"""Offline tests for digest formatting, one-way alerts, and interactive CLI review."""

import asyncio
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from doot.cli import app
from doot.config import DootSettings
from doot.db import DootRepository, initialize_schema, open_connection
from doot.models import JobStatus, ScoreResult
from doot.notify import build_digest_text, dispatch_one_way_alert
from tests.conftest import make_sample_job


async def _seed_scored_jobs(repo: DootRepository) -> tuple[int, int]:
    j1 = make_sample_job(external_job_id="top-1", alert_only=False)
    j1.title = 'Senior "Cloud" Engineer'
    j2 = make_sample_job(external_job_id="second-2", alert_only=True)
    j2.title = "Backend Node Developer"

    id1, _ = await repo.upsert_job(j1)
    id2, _ = await repo.upsert_job(j2)

    await repo.save_job_score(
        id1,
        ScoreResult(
            score=5,
            should_apply=True,
            reasoning="Strong AWS Lambda fit.",
            missing_skills=["Go"],
        ),
        score_cache_key="k1",
    )
    await repo.save_job_score(
        id2,
        ScoreResult(
            score=4,
            should_apply=True,
            reasoning="Good Node.js fit.",
            missing_skills=[],
        ),
        score_cache_key="k2",
    )
    return id1, id2


async def test_digest_query_ranking_and_mark_notified(repo: DootRepository) -> None:
    id1, id2 = await _seed_scored_jobs(repo)

    ranked = await repo.list_jobs_by_statuses((JobStatus.SCORED,), min_score=4, limit=10)
    assert [j.id for j in ranked] == [id1, id2]

    text = build_digest_text(ranked)
    assert "[5/5] Acme Corp" in text
    assert "Gaps: Go" in text
    assert "alert-only" in text

    changed = await repo.mark_jobs_notified([id1, id2])
    assert changed == 2
    j1 = await repo.get_job(id1)
    assert j1 is not None and j1.status == JobStatus.NOTIFIED


@respx.mock
async def test_one_way_alerts_macos_and_webhook(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    await _seed_scored_jobs(repo)
    jobs = await repo.list_jobs_by_statuses((JobStatus.SCORED,), min_score=4)

    # 1) macOS osascript with quote escaping
    captured_cmds: list[list[str]] = []

    async def _fake_runner(cmd: list[str]) -> int:
        captured_cmds.append(cmd)
        return 0

    sent_macos = await dispatch_one_way_alert(tmp_settings, jobs, runner=_fake_runner)
    assert sent_macos is True
    assert captured_cmds[0][0] == "osascript"
    assert '\\"Cloud\\"' in captured_cmds[0][2]

    # 2) Webhook POST
    route = respx.post("https://hooks.example.test/doot").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    wh_settings = tmp_settings.model_copy(
        update={"notify_method": "webhook", "notify_webhook_url": "https://hooks.example.test/doot"}
    )
    async with httpx.AsyncClient() as client:
        sent_wh = await dispatch_one_way_alert(wh_settings, jobs, client=client)
    assert sent_wh is True
    assert route.called


def test_cli_digest_and_interactive_review_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "review_data"
    db_path = data_dir / "doot.db"

    async def _setup() -> tuple[int, int]:
        async with open_connection(db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await _seed_scored_jobs(repo)

    id1, id2 = asyncio.run(_setup())

    monkeypatch.setenv("DOOT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DOOT_DB_PATH", str(db_path))
    monkeypatch.setenv("DOOT_NOTIFY_METHOD", "none")

    runner = CliRunner()

    # Digest prints table and transitions SCORED -> NOTIFIED
    res_digest = runner.invoke(app, ["digest", "--min-score", "4"])
    assert res_digest.exit_code == 0
    assert "Doot Ranked Digest" in res_digest.output
    assert "Transitioned 2 SCORED job(s) to NOTIFIED." in res_digest.output

    # Review: apply to first job ('a'), skip second job ('s')
    res_review = runner.invoke(app, ["review", "--min-score", "4"], input="a\ns\n")
    assert res_review.exit_code == 0
    assert f"Job {id1} marked APPLIED" in res_review.output
    assert f"Job {id2} marked SKIPPED" in res_review.output

    async def _verify() -> tuple[JobStatus, JobStatus]:
        async with open_connection(db_path) as conn:
            repo = DootRepository(conn)
            j1 = await repo.get_job(id1)
            j2 = await repo.get_job(id2)
            assert j1 is not None and j2 is not None
            return j1.status, j2.status

    s1, s2 = asyncio.run(_verify())
    assert s1 == JobStatus.APPLIED
    assert s2 == JobStatus.SKIPPED
