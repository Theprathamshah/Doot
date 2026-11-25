"""Tests for safety guardrails, caps, kill switch, and CLI commands."""

from pathlib import Path

import aiosqlite
import pytest
from typer.testing import CliRunner

from doot.cli import app
from doot.config import DootSettings
from doot.db import DootRepository
from doot.kill_switch import SafetyGuardError, assert_not_stopped, check_apply_guardrails
from doot.models import ATSProvider, JobStatus
from tests.conftest import make_sample_job


def test_kill_switch_blocks_when_stop_file_exists(tmp_settings: DootSettings) -> None:
    assert_not_stopped(tmp_settings)
    tmp_settings.stop_file_path.write_text("stop", encoding="utf-8")

    with pytest.raises(SafetyGuardError, match="Global kill switch active"):
        assert_not_stopped(tmp_settings)


async def test_apply_guardrails_enforce_dry_run_and_caps(
    tmp_settings: DootSettings,
    db_conn: aiosqlite.Connection,
    repo: DootRepository,
) -> None:
    company_id = await repo.upsert_company(
        name="Acme Corp",
        normalized_name="acme corp",
        ats_provider=ATSProvider.GREENHOUSE,
        board_token="acme",
    )

    # Dry-run blocks even if apply_flag=False or dry_run=True
    with pytest.raises(SafetyGuardError, match="dry-run mode is active"):
        await check_apply_guardrails(tmp_settings, repo, company_id=company_id, apply_flag=True)

    # Enable live apply mode with tight caps for testing
    live_settings = tmp_settings.model_copy(
        update={"dry_run": False, "daily_apply_cap": 2, "per_company_cap": 1}
    )
    await check_apply_guardrails(live_settings, repo, company_id=company_id, apply_flag=True)

    # Record 1 automated application for Acme -> triggers per-company cap of 1
    job_id, _ = await repo.upsert_job(
        make_sample_job(company_id=company_id, external_job_id="gh-1")
    )
    await db_conn.execute(
        """
        INSERT INTO applications (job_id, status, submitted_at)
        VALUES (?, 'APPLIED', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        """,
        (job_id,),
    )
    await db_conn.commit()

    with pytest.raises(SafetyGuardError, match="Per-company application cap reached"):
        await check_apply_guardrails(live_settings, repo, company_id=company_id, apply_flag=True)


def test_cli_init_status_stats_and_stop_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "cli_data"
    monkeypatch.setenv("DOOT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DOOT_DB_PATH", str(data_dir / "doot.db"))

    runner = CliRunner()
    res_init = runner.invoke(app, ["init"])
    assert res_init.exit_code == 0
    assert (data_dir / "doot.db").exists()

    res_status = runner.invoke(app, ["status"])
    assert res_status.exit_code == 0
    assert "Doot Status" in res_status.output

    res_stats = runner.invoke(app, ["stats"])
    assert res_stats.exit_code == 0
    assert JobStatus.DISCOVERED.value in res_stats.output

    res_stop = runner.invoke(app, ["stop"])
    assert res_stop.exit_code == 0
    assert (data_dir / "STOP").exists()

    res_resume = runner.invoke(app, ["resume"])
    assert res_resume.exit_code == 0
    assert not (data_dir / "STOP").exists()
