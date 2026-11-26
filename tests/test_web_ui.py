"""Offline tests for the local Starlette web dashboard and JSON API."""

import asyncio
from typing import Any

from starlette.testclient import TestClient

from doot.config import DootSettings
from doot.db import DootRepository, initialize_schema, open_connection
from doot.models import (
    JobPosting,
    JobStatus,
    ProfileLinks,
    ScoreResult,
    TailoredAssets,
    UserProfile,
)
from doot.web import create_web_app
from tests.conftest import make_sample_job


def _seed_web_db(settings: DootSettings) -> tuple[int, int, int]:
    profile = UserProfile(
        full_name="Pratham Shah",
        email="pratham@example.com",
        phone="+91-9876543210",
        location="Ahmedabad, India",
        work_authorization_india="Authorized in India",
        years_of_experience=2.5,
        current_ctc_lpa=12.0,
        expected_ctc_lpa=22.0,
        notice_period_days=30,
        links=ProfileLinks(linkedin="https://linkedin.com/in/pratham"),
    )
    (settings.data_dir / "profile.json").write_text(profile.model_dump_json(), encoding="utf-8")
    (settings.data_dir / "stories.md").write_text(
        "### [story:lambda-30x] AWS Lambda\nCut latency 30x via query batching.\n",
        encoding="utf-8",
    )

    async def _seed() -> tuple[int, int, int]:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)

            j_sel = make_sample_job(external_job_id="web-sel-1")
            j_sel.title = "Senior AWS Backend Engineer"
            j_sel.jd_text = "Build AWS Lambda and Spring Boot microservices."
            j_sel.is_remote = True
            j_sel.salary_raw = "30 LPA"
            id_sel, _ = await repo.upsert_job(j_sel)
            await repo.save_job_score(
                id_sel,
                ScoreResult(
                    score=5,
                    should_apply=True,
                    reasoning="Strong match",
                    missing_skills=[],
                ),
                score_cache_key="wk1",
            )

            j_low = make_sample_job(external_job_id="web-low-2", source="jobicy", alert_only=True)
            j_low.title = "Junior PHP Maintainer"
            j_low.apply_url = "javascript:alert(1)"  # Must be sanitized by API
            id_low, _ = await repo.upsert_job(j_low)
            await repo.save_job_score(
                id_low,
                ScoreResult(score=2, should_apply=False, reasoning="Weak match", missing_skills=[]),
                score_cache_key="wk2",
            )

            j_filt = make_sample_job(external_job_id="web-filt-3")
            id_filt, _ = await repo.upsert_job(j_filt)
            await repo.mark_job_filtered_out(id_filt, "US only")
            return id_sel, id_low, id_filt

    return asyncio.run(_seed())


def test_web_ui_endpoints_filters_xss_url_guard_and_actions(tmp_settings: DootSettings) -> None:
    id_sel, id_low, _id_filt = _seed_web_db(tmp_settings)

    class _FakeWebLLM:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"

        async def score_job(self, redacted_profile: str, untrusted_jd: str) -> ScoreResult:
            raise NotImplementedError

        async def draft_tailoring(
            self, evidence_chunks: list[str], job: JobPosting
        ) -> TailoredAssets:
            return TailoredAssets(
                tailored_summary="Cut AWS Lambda latency 30x.",
                cover_note="Strong AWS Lambda + Java experience.",
                outreach_dm="Hi, saw the AWS Backend role.",
                cited_story_ids=["lambda-30x"],
            )

        async def draft_free_text_answer(
            self, question: str, evidence_chunks: list[str], job: JobPosting
        ) -> str:
            raise NotImplementedError

    client = TestClient(create_web_app(tmp_settings, llm_provider=_FakeWebLLM()))

    # 1) HTML dashboard loads with color-scheme meta & semantic table
    home = client.get("/")
    assert home.status_code == 200
    assert '<meta name="color-scheme" content="light dark"' in home.text
    assert "Doot — Job Triage & Direct Apply" in home.text

    # 2) Stats endpoint
    stats = client.get("/api/stats").json()
    assert stats["selected_count"] == 1
    assert stats["total_jobs"] == 3

    # 3) Selected view shows only 4-5/5 actionable jobs
    sel = client.get("/api/jobs?view=selected").json()
    assert sel["total"] == 1
    assert sel["jobs"][0]["id"] == id_sel
    assert sel["jobs"][0]["apply_url"].startswith("https://")

    # 4) All view shows all 3 jobs & strips non-http(s) apply_url
    all_jobs = client.get("/api/jobs?view=all").json()
    assert all_jobs["total"] == 3
    low_row: dict[str, Any] = next(j for j in all_jobs["jobs"] if j["id"] == id_low)
    assert low_row["apply_url"] == ""

    # 5) Detail + RAG Tailor + Status transition to APPLIED
    detail = client.get(f"/api/jobs/{id_sel}").json()
    assert "AWS Lambda" in detail["jd_text"]

    tailor_res = client.post(f"/api/jobs/{id_sel}/tailor")
    assert tailor_res.status_code == 200
    assert tailor_res.json()["cited_story_ids"] == ["lambda-30x"]

    applied_res = client.post(f"/api/jobs/{id_sel}/status", json={"status": "APPLIED"})
    assert applied_res.status_code == 200
    assert applied_res.json()["status"] == JobStatus.APPLIED.value

    # Cannot transition APPLIED -> SKIPPED (409 conflict)
    conflict = client.post(f"/api/jobs/{id_sel}/status", json={"status": "SKIPPED"})
    assert conflict.status_code == 409

    # 6) Portal search & stats portals list
    stats_after = client.get("/api/stats").json()
    portal_names = {p["portal"] for p in stats_after["portals"]}
    assert "jobicy" in portal_names
    assert "greenhouse" in portal_names

    jobicy_only = client.get("/api/jobs?view=all&portal=jobicy").json()
    assert jobicy_only["total"] == 1
    assert jobicy_only["jobs"][0]["id"] == id_low

    # 7) Sorting by added time (discovered_at) & extra filters (remote / unscored)
    asc_added = client.get("/api/jobs?view=all&sort_by=discovered_at&sort_dir=asc").json()
    desc_added = client.get("/api/jobs?view=all&sort_by=discovered_at&sort_dir=desc").json()
    assert [j["id"] for j in asc_added["jobs"]] == list(
        reversed([j["id"] for j in desc_added["jobs"]])
    )

    remote_only = client.get("/api/jobs?view=all&remote=true").json()
    assert remote_only["total"] == 1
    assert remote_only["jobs"][0]["id"] == id_sel

    onsite_only = client.get("/api/jobs?view=all&remote=false").json()
    assert onsite_only["total"] == 2

    salary_only = client.get("/api/jobs?view=all&has_salary=true").json()
    assert salary_only["total"] == 1
    assert salary_only["jobs"][0]["id"] == id_sel

    # 8) Delete job posting from UI API
    assert "confirm-delete-dialog" in home.text
    assert 'commandfor="jobs-table"' in home.text

    del_res = client.delete(f"/api/jobs/{id_low}")
    assert del_res.status_code == 200
    assert del_res.json() == {"deleted": True, "id": id_low}
    assert client.get(f"/api/jobs/{id_low}").status_code == 404
    assert client.delete(f"/api/jobs/{id_low}").status_code == 404

    stats_final = client.get("/api/stats").json()
    assert stats_final["total_jobs"] == 2
