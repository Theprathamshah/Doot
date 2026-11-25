"""Shared offline pytest fixtures for Doot."""

from collections.abc import AsyncIterator
from pathlib import Path

import aiosqlite
import pytest

from doot.config import DootSettings
from doot.db import DootRepository, initialize_schema, open_connection
from doot.models import ATSProvider, JobPosting


@pytest.fixture
def tmp_settings(tmp_path: Path) -> DootSettings:
    """Return isolated DootSettings backed by a temporary directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return DootSettings(
        env="test",
        data_dir=data_dir,
        db_path=data_dir / "doot.db",
        gemini_api_key="test-secret-gemini-key",
    )


@pytest.fixture
async def db_conn(tmp_settings: DootSettings) -> AsyncIterator[aiosqlite.Connection]:
    """Provide an initialized SQLite WAL connection for tests."""
    async with open_connection(tmp_settings.db_path) as conn:
        await initialize_schema(conn)
        yield conn


@pytest.fixture
async def repo(db_conn: aiosqlite.Connection) -> DootRepository:
    """Provide a repository bound to an initialized test database."""
    return DootRepository(db_conn)


def make_sample_job(
    *,
    external_job_id: str = "gh-12345",
    company_id: int | None = None,
    source: str = "greenhouse",
    alert_only: bool = False,
) -> JobPosting:
    """Create a valid sample JobPosting for unit tests."""
    return JobPosting(
        company_id=company_id,
        source=source,
        external_job_id=external_job_id,
        ats_provider=ATSProvider.GREENHOUSE if source == "greenhouse" else None,
        board_token="acme" if source == "greenhouse" else None,
        canonical_url=f"https://boards.greenhouse.io/acme/jobs/{external_job_id}",
        apply_url=f"https://boards.greenhouse.io/acme/jobs/{external_job_id}#app",
        title="Backend Software Engineer (AWS)",
        normalized_title="backend software engineer aws",
        company_name="Acme Corp",
        normalized_company="acme corp",
        location_raw="Bengaluru, India (Hybrid)",
        is_remote=False,
        india_eligible=True,
        alert_only=alert_only,
        jd_text="Looking for 2+ years Java, Spring Boot, AWS Lambda experience.",
        jd_hash="abc123",
        dedupe_cluster_id="cluster-acme-backend",
    )
