"""Tests for SQLite WAL pragmas, repository operations, and state transitions."""

import aiosqlite
import pytest

from doot.db import DootRepository, InvalidStateTransitionError
from doot.models import ALLOWED_TRANSITIONS, ATSProvider, CompanySource, JobStatus
from tests.conftest import make_sample_job


async def test_wal_and_foreign_keys_enabled(db_conn: aiosqlite.Connection) -> None:
    cursor = await db_conn.execute("PRAGMA journal_mode;")
    row = await cursor.fetchone()
    assert row is not None
    assert str(row[0]).lower() == "wal"

    fk_cursor = await db_conn.execute("PRAGMA foreign_keys;")
    fk_row = await fk_cursor.fetchone()
    assert fk_row is not None
    assert int(fk_row[0]) == 1


async def test_upsert_job_creates_initial_event_and_preserves_status(repo: DootRepository) -> None:
    company_id = await repo.upsert_company(
        name="Acme Corp",
        normalized_name="acme corp",
        ats_provider=ATSProvider.GREENHOUSE,
        board_token="acme",
        source=CompanySource.MANUAL,
    )
    job_id, created = await repo.upsert_job(make_sample_job(company_id=company_id))
    assert created is True

    events = await repo.get_job_events(job_id)
    assert len(events) == 1
    assert events[0]["from_status"] is None
    assert events[0]["to_status"] == JobStatus.DISCOVERED.value

    await repo.transition_job(job_id, JobStatus.SCORED, actor="scorer", reason="score=5")
    await repo.transition_job(job_id, JobStatus.NOTIFIED, actor="digest", reason="in digest")

    # Re-upserting the same job must update metadata without resetting status
    updated_job = make_sample_job(company_id=company_id)
    updated_job.location_raw = "Remote - India"
    second_id, second_created = await repo.upsert_job(updated_job)
    assert second_id == job_id
    assert second_created is False

    stored = await repo.get_job(job_id)
    assert stored is not None
    assert stored.status == JobStatus.NOTIFIED
    assert stored.location_raw == "Remote - India"


async def test_valid_and_invalid_state_transitions(repo: DootRepository) -> None:
    job_id, _ = await repo.upsert_job(make_sample_job(external_job_id="gh-999"))

    # Cannot jump directly from DISCOVERED to APPLIED
    with pytest.raises(InvalidStateTransitionError):
        await repo.transition_job(job_id, JobStatus.APPLIED, actor="test")

    # Alert-only path: DISCOVERED -> SCORED -> NOTIFIED -> APPLIED
    await repo.transition_job(job_id, JobStatus.SCORED, actor="scorer")
    await repo.transition_job(job_id, JobStatus.NOTIFIED, actor="digest")
    applied = await repo.transition_job(job_id, JobStatus.APPLIED, actor="cli")
    assert applied.status == JobStatus.APPLIED

    # APPLIED is strictly terminal
    assert ALLOWED_TRANSITIONS[JobStatus.APPLIED] == frozenset()
    with pytest.raises(InvalidStateTransitionError):
        await repo.transition_job(job_id, JobStatus.SKIPPED, actor="cli")

    events = await repo.get_job_events(job_id)
    assert [e["to_status"] for e in events] == [
        "DISCOVERED",
        "SCORED",
        "NOTIFIED",
        "APPLIED",
    ]


async def test_qa_cache_company_overrides_global(repo: DootRepository) -> None:
    company_id = await repo.upsert_company(
        name="Acme Corp",
        normalized_name="acme corp",
        ats_provider=ATSProvider.LEVER,
        board_token="acme",
    )
    await repo.upsert_qa_cache(
        question_normalized="notice period days",
        question_sample="What is your notice period in days?",
        answer="30",
        scope="global",
    )
    await repo.upsert_qa_cache(
        question_normalized="notice period days",
        question_sample="Notice period?",
        answer="15 (negotiable)",
        scope="company",
        company_id=company_id,
    )

    assert (
        await repo.lookup_qa_cache("notice period days", company_id=company_id)
        == "15 (negotiable)"
    )
    assert await repo.lookup_qa_cache("notice period days", company_id=9999) == "30"
