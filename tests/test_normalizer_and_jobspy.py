"""Offline tests for URL/location normalization and JobSpy timeout handling."""

import time

from doot.collectors.jobspy_collector import JobSpyCollector
from doot.models import RawJob
from doot.pipeline.normalizer import (
    canonicalize_url,
    infer_remote_and_india,
    normalize_raw_job,
    normalize_text,
)


def test_canonicalize_url_strips_tracking_params() -> None:
    raw = "https://Boards.Greenhouse.io/acme/jobs/1001/?utm_source=linkedin&gh_src=abc&keep=1#app"
    assert canonicalize_url(raw) == "https://boards.greenhouse.io/acme/jobs/1001?keep=1"


def test_normalize_text_collapses_indian_city_aliases() -> None:
    assert normalize_text("Backend Engineer, Bangalore!") == normalize_text(
        "Backend Engineer - Bengaluru"
    )


def test_infer_remote_and_india_eligibility() -> None:
    assert infer_remote_and_india("Bengaluru, India") == (False, True)
    assert infer_remote_and_india("Remote - India") == (True, True)
    assert infer_remote_and_india("Remote") == (True, True)
    assert infer_remote_and_india("Remote (Worldwide)") == (True, True)
    assert infer_remote_and_india("Remote (US Only)") == (True, False)
    assert infer_remote_and_india("Remote (United States)") == (True, False)
    assert infer_remote_and_india("Remote (Canada)") == (True, False)
    assert infer_remote_and_india("San Francisco, CA") == (False, False)


def test_normalize_raw_job_builds_cluster_and_hash() -> None:
    raw = RawJob(
        source="greenhouse",
        external_job_id="1001",
        title="Backend Engineer",
        company_name="Acme Corp",
        canonical_url="https://boards.greenhouse.io/acme/jobs/1001?utm_source=x",
        apply_url="https://boards.greenhouse.io/acme/jobs/1001#app",
        location_raw="Bangalore, India (Remote)",
        jd_text="AWS   Lambda   Java",
    )
    posting = normalize_raw_job(raw, company_id=5)
    assert posting.is_remote is True
    assert posting.india_eligible is True
    assert posting.normalized_company == "acme corp"
    assert len(posting.jd_hash) == 64
    assert len(posting.dedupe_cluster_id) == 16


async def test_jobspy_collector_handles_timeout_and_alert_only_flag() -> None:
    def _slow_or_fast(**kwargs: object) -> list[dict[str, object]]:
        if kwargs.get("search_term") == "slow":
            time.sleep(0.3)
        return [
            {
                "site": "naukri",
                "id": "nk-1",
                "title": "Java AWS Developer",
                "company": "FinTech IN",
                "location": "Bengaluru",
                "job_url": "https://www.naukri.com/job-listings-1",
                "description": "Java Spring Boot AWS",
            }
        ]

    collector = JobSpyCollector(
        sites=["naukri"],
        search_terms=["slow", "fast"],
        timeout_seconds=0.05,
        scrape_fn=_slow_or_fast,
    )
    jobs = await collector.fetch()
    assert len(jobs) == 1
    assert jobs[0].source == "jobspy_naukri"
    assert jobs[0].alert_only is True
