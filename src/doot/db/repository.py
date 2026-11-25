"""Persistence repository with enforced state transitions and audit logging."""

import json
from typing import Any

import aiosqlite

from doot.models.enums import (
    ALLOWED_TRANSITIONS,
    ATSProvider,
    CompanySource,
    JobStatus,
    RunStatus,
    RunType,
)
from doot.models.job import CompanyRecord, JobPosting, JobRecord, ScoreResult


class InvalidStateTransitionError(ValueError):
    """Raised when a job status transition violates ALLOWED_TRANSITIONS."""


def _row_to_job(row: aiosqlite.Row) -> JobRecord:
    missing_skills_raw = row["missing_skills_json"]
    missing_skills: list[str] = json.loads(missing_skills_raw) if missing_skills_raw else []
    ats_raw = row["ats_provider"]
    return JobRecord(
        id=int(row["id"]),
        company_id=int(row["company_id"]) if row["company_id"] is not None else None,
        source=str(row["source"]),
        external_job_id=str(row["external_job_id"]),
        ats_provider=ATSProvider(ats_raw) if ats_raw else None,
        board_token=row["board_token"],
        canonical_url=str(row["canonical_url"]),
        apply_url=str(row["apply_url"]),
        title=str(row["title"]),
        normalized_title=str(row["normalized_title"]),
        company_name=str(row["company_name"]),
        normalized_company=str(row["normalized_company"]),
        location_raw=str(row["location_raw"]),
        is_remote=bool(row["is_remote"]),
        india_eligible=bool(row["india_eligible"]),
        alert_only=bool(row["alert_only"]),
        salary_raw=row["salary_raw"],
        jd_text=str(row["jd_text"]),
        jd_hash=str(row["jd_hash"]),
        dedupe_cluster_id=str(row["dedupe_cluster_id"]),
        status=JobStatus(str(row["status"])),
        rule_filter_reason=row["rule_filter_reason"],
        llm_score=row["llm_score"],
        llm_reasoning=row["llm_reasoning"],
        missing_skills=missing_skills,
        should_apply=(bool(row["should_apply"]) if row["should_apply"] is not None else None),
        score_cache_key=row["score_cache_key"],
        discovered_at=str(row["discovered_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_company(row: aiosqlite.Row) -> CompanyRecord:
    return CompanyRecord(
        id=int(row["id"]),
        name=str(row["name"]),
        normalized_name=str(row["normalized_name"]),
        ats_provider=ATSProvider(str(row["ats_provider"])),
        board_token=str(row["board_token"]),
        careers_url=row["careers_url"],
        source=CompanySource(str(row["source"])),
        is_blocklisted=bool(row["is_blocklisted"]),
        is_active=bool(row["is_active"]),
        consecutive_failures=int(row["consecutive_failures"]),
        last_crawled_at=row["last_crawled_at"],
    )


class DootRepository:
    """Async repository encapsulating all SQLite operations and state invariants."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def upsert_company(
        self,
        *,
        name: str,
        normalized_name: str,
        ats_provider: ATSProvider,
        board_token: str,
        careers_url: str | None = None,
        source: CompanySource = CompanySource.MANUAL,
        is_blocklisted: bool = False,
    ) -> int:
        """Insert or update a company watchlist entry and return its ID."""
        await self.conn.execute(
            """
            INSERT INTO companies (
                name, normalized_name, ats_provider, board_token,
                careers_url, source, is_blocklisted
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ats_provider, board_token) DO UPDATE SET
                name = excluded.name,
                normalized_name = excluded.normalized_name,
                careers_url = COALESCE(excluded.careers_url, companies.careers_url),
                is_blocklisted = excluded.is_blocklisted,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            (
                name,
                normalized_name,
                ats_provider.value,
                board_token,
                careers_url,
                source.value,
                int(is_blocklisted),
            ),
        )
        cursor = await self.conn.execute(
            "SELECT id FROM companies WHERE ats_provider = ? AND board_token = ?",
            (ats_provider.value, board_token),
        )
        row = await cursor.fetchone()
        await self.conn.commit()
        assert row is not None
        return int(row["id"])

    async def list_companies(self, *, active_only: bool = True) -> list[CompanyRecord]:
        """List companies from the watchlist."""
        query = "SELECT * FROM companies"
        if active_only:
            query += " WHERE is_active = 1 AND is_blocklisted = 0"
        query += " ORDER BY name ASC"
        cursor = await self.conn.execute(query)
        rows = await cursor.fetchall()
        return [_row_to_company(r) for r in rows]

    async def get_company_by_id(self, company_id: int) -> CompanyRecord | None:
        """Fetch a company by primary key."""
        cursor = await self.conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,))
        row = await cursor.fetchone()
        return _row_to_company(row) if row is not None else None

    async def get_company_by_ats(
        self,
        ats_provider: ATSProvider,
        board_token: str,
    ) -> CompanyRecord | None:
        """Fetch a company by ATS provider and board token."""
        cursor = await self.conn.execute(
            "SELECT * FROM companies WHERE ats_provider = ? AND board_token = ?",
            (ats_provider.value, board_token),
        )
        row = await cursor.fetchone()
        return _row_to_company(row) if row is not None else None

    async def record_company_crawl_success(self, company_id: int) -> None:
        """Reset failure streak and update last_crawled_at after a successful fetch."""
        await self.conn.execute(
            """
            UPDATE companies
            SET consecutive_failures = 0,
                last_crawled_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (company_id,),
        )
        await self.conn.commit()

    async def record_company_crawl_failure(
        self,
        company_id: int,
        *,
        max_failures: int = 5,
    ) -> bool:
        """Increment failure counter and auto-pause the board once threshold is hit.

        Returns True if the board was deactivated.
        """
        cursor = await self.conn.execute(
            "SELECT consecutive_failures FROM companies WHERE id = ?",
            (company_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        new_failures = int(row["consecutive_failures"]) + 1
        deactivated = new_failures >= max_failures
        await self.conn.execute(
            """
            UPDATE companies
            SET consecutive_failures = ?,
                is_active = CASE WHEN ? THEN 0 ELSE is_active END,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (new_failures, int(deactivated), company_id),
        )
        await self.conn.commit()
        return deactivated

    async def upsert_job(self, job: JobPosting, *, actor: str = "system") -> tuple[int, bool]:
        """Insert a new job (with initial DISCOVERED event) or refresh an existing one.

        Returns `(job_id, created)`. Existing job statuses are preserved on re-crawl.
        """
        existing_cursor = await self.conn.execute(
            "SELECT id FROM jobs WHERE source = ? AND external_job_id = ?",
            (job.source, job.external_job_id),
        )
        existing = await existing_cursor.fetchone()

        if existing is not None:
            job_id = int(existing["id"])
            await self.conn.execute(
                """
                UPDATE jobs SET
                    company_id = COALESCE(?, company_id),
                    ats_provider = COALESCE(?, ats_provider),
                    board_token = COALESCE(?, board_token),
                    canonical_url = ?,
                    apply_url = ?,
                    title = ?,
                    normalized_title = ?,
                    company_name = ?,
                    normalized_company = ?,
                    location_raw = ?,
                    is_remote = ?,
                    india_eligible = ?,
                    alert_only = ?,
                    salary_raw = COALESCE(?, salary_raw),
                    jd_text = ?,
                    jd_hash = ?,
                    dedupe_cluster_id = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (
                    job.company_id,
                    job.ats_provider.value if job.ats_provider else None,
                    job.board_token,
                    job.canonical_url,
                    job.apply_url,
                    job.title,
                    job.normalized_title,
                    job.company_name,
                    job.normalized_company,
                    job.location_raw,
                    int(job.is_remote),
                    int(job.india_eligible),
                    int(job.alert_only),
                    job.salary_raw,
                    job.jd_text,
                    job.jd_hash,
                    job.dedupe_cluster_id,
                    job_id,
                ),
            )
            await self.conn.commit()
            return job_id, False

        cursor = await self.conn.execute(
            """
            INSERT INTO jobs (
                company_id, source, external_job_id, ats_provider, board_token,
                canonical_url, apply_url, title, normalized_title,
                company_name, normalized_company, location_raw,
                is_remote, india_eligible, alert_only, salary_raw,
                jd_text, jd_hash, dedupe_cluster_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.company_id,
                job.source,
                job.external_job_id,
                job.ats_provider.value if job.ats_provider else None,
                job.board_token,
                job.canonical_url,
                job.apply_url,
                job.title,
                job.normalized_title,
                job.company_name,
                job.normalized_company,
                job.location_raw,
                int(job.is_remote),
                int(job.india_eligible),
                int(job.alert_only),
                job.salary_raw,
                job.jd_text,
                job.jd_hash,
                job.dedupe_cluster_id,
                JobStatus.DISCOVERED.value,
            ),
        )
        assert cursor.lastrowid is not None
        job_id = int(cursor.lastrowid)
        await self.conn.execute(
            """
            INSERT INTO job_events (job_id, from_status, to_status, actor, reason)
            VALUES (?, NULL, ?, ?, ?)
            """,
            (job_id, JobStatus.DISCOVERED.value, actor, "initial discovery"),
        )
        await self.conn.commit()
        return job_id, True

    async def get_job(self, job_id: int) -> JobRecord | None:
        """Fetch a single job by primary key."""
        cursor = await self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        return _row_to_job(row) if row is not None else None

    async def get_job_by_ats_key(
        self,
        ats_provider: ATSProvider,
        board_token: str,
        external_job_id: str,
    ) -> JobRecord | None:
        """Fetch a job matching an ATS identity regardless of discovery source."""
        cursor = await self.conn.execute(
            """
            SELECT * FROM jobs
            WHERE ats_provider = ? AND board_token = ? AND external_job_id = ?
            LIMIT 1
            """,
            (ats_provider.value, board_token, external_job_id),
        )
        row = await cursor.fetchone()
        return _row_to_job(row) if row is not None else None

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        """List jobs optionally filtered by status."""
        if status is not None:
            cursor = await self.conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY COALESCE(llm_score, 0) DESC, id DESC
                LIMIT ?
                """,
                (status.value, limit),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM jobs ORDER BY COALESCE(llm_score, 0) DESC, id DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [_row_to_job(r) for r in rows]

    async def list_jobs_by_statuses(
        self,
        statuses: tuple[JobStatus, ...],
        *,
        min_score: int = 1,
        limit: int = 50,
    ) -> list[JobRecord]:
        """List jobs matching any of the given statuses and minimum LLM score."""
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        params: list[Any] = [s.value for s in statuses]
        params.extend([min_score, limit])
        cursor = await self.conn.execute(
            f"""
            SELECT * FROM jobs
            WHERE status IN ({placeholders})
              AND COALESCE(llm_score, 0) >= ?
            ORDER BY COALESCE(llm_score, 0) DESC, alert_only ASC, id DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [_row_to_job(r) for r in rows]

    async def mark_jobs_notified(
        self,
        job_ids: list[int],
        *,
        actor: str = "digest",
    ) -> int:
        """Transition SCORED jobs to NOTIFIED and return the number transitioned."""
        count = 0
        for job_id in job_ids:
            job = await self.get_job(job_id)
            if job is not None and job.status == JobStatus.SCORED:
                await self.transition_job(
                    job_id,
                    JobStatus.NOTIFIED,
                    actor=actor,
                    reason="included in digest",
                )
                count += 1
        return count

    async def transition_job(
        self,
        job_id: int,
        to_status: JobStatus,
        *,
        actor: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Atomically validate and apply a job state transition with audit logging."""
        cursor = await self.conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            raise LookupError(f"Job {job_id} not found")

        current_status = JobStatus(str(row["status"]))
        allowed = ALLOWED_TRANSITIONS[current_status]
        if to_status not in allowed:
            raise InvalidStateTransitionError(
                f"Transition {current_status.value} -> {to_status.value} "
                f"is not allowed for job {job_id}"
            )

        await self.conn.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (to_status.value, job_id),
        )
        await self.conn.execute(
            """
            INSERT INTO job_events (job_id, from_status, to_status, actor, reason, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                current_status.value,
                to_status.value,
                actor,
                reason,
                json.dumps(metadata) if metadata is not None else None,
            ),
        )
        await self.conn.commit()
        updated = await self.get_job(job_id)
        assert updated is not None
        return updated

    async def get_cached_score(self, score_cache_key: str) -> ScoreResult | None:
        """Return a previously computed ScoreResult matching score_cache_key."""
        cursor = await self.conn.execute(
            """
            SELECT llm_score, llm_reasoning, missing_skills_json, should_apply, india_eligible
            FROM jobs
            WHERE score_cache_key = ?
              AND llm_score IS NOT NULL
              AND llm_reasoning IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (score_cache_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        missing_raw = row["missing_skills_json"]
        return ScoreResult(
            score=int(row["llm_score"]),
            should_apply=bool(row["should_apply"]),
            reasoning=str(row["llm_reasoning"]),
            missing_skills=json.loads(missing_raw) if missing_raw else [],
            india_remote_eligible=bool(row["india_eligible"]),
        )

    async def mark_job_filtered_out(
        self,
        job_id: int,
        reason: str,
        *,
        actor: str = "rule_filter",
    ) -> JobRecord:
        """Persist filter rejection reason and transition job to FILTERED_OUT."""
        await self.conn.execute(
            "UPDATE jobs SET rule_filter_reason = ? WHERE id = ?",
            (reason, job_id),
        )
        return await self.transition_job(
            job_id,
            JobStatus.FILTERED_OUT,
            actor=actor,
            reason=reason,
        )

    async def save_job_score(
        self,
        job_id: int,
        result: ScoreResult,
        *,
        score_cache_key: str,
        actor: str = "scorer",
    ) -> JobRecord:
        """Persist LLM ScoreResult and transition job to SCORED."""
        await self.conn.execute(
            """
            UPDATE jobs
            SET llm_score = ?,
                llm_reasoning = ?,
                missing_skills_json = ?,
                should_apply = ?,
                india_eligible = ?,
                score_cache_key = ?
            WHERE id = ?
            """,
            (
                result.score,
                result.reasoning,
                json.dumps(result.missing_skills),
                int(result.should_apply),
                int(result.india_remote_eligible),
                score_cache_key,
                job_id,
            ),
        )
        return await self.transition_job(
            job_id,
            JobStatus.SCORED,
            actor=actor,
            reason=f"score={result.score}",
            metadata={"should_apply": result.should_apply, "cache_key": score_cache_key},
        )

    async def get_job_events(self, job_id: int) -> list[dict[str, Any]]:
        """Return chronological audit events for a job."""
        cursor = await self.conn.execute(
            "SELECT * FROM job_events WHERE job_id = ? ORDER BY id ASC",
            (job_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def count_automated_applies_today(self) -> int:
        """Count automated applications submitted today (UTC date)."""
        cursor = await self.conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM applications
            WHERE status = 'APPLIED'
              AND submitted_at IS NOT NULL
              AND date(submitted_at) = date('now')
            """
        )
        row = await cursor.fetchone()
        return int(row["cnt"]) if row is not None else 0

    async def count_automated_applies_for_company(self, company_id: int) -> int:
        """Count automated applications submitted for a given company."""
        cursor = await self.conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            WHERE a.status = 'APPLIED'
              AND a.submitted_at IS NOT NULL
              AND j.company_id = ?
            """,
            (company_id,),
        )
        row = await cursor.fetchone()
        return int(row["cnt"]) if row is not None else 0

    async def upsert_qa_cache(
        self,
        *,
        question_normalized: str,
        question_sample: str,
        answer: str,
        scope: str = "global",
        company_id: int | None = None,
        confirmed_by_user: bool = True,
    ) -> None:
        """Store or update a confirmed custom ATS question answer."""
        if scope == "global":
            company_id = None
        await self.conn.execute(
            """
            INSERT INTO qa_cache (
                question_normalized, question_sample, answer, scope, company_id, confirmed_by_user
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_normalized, scope, COALESCE(company_id, -1)) DO UPDATE SET
                answer = excluded.answer,
                question_sample = excluded.question_sample,
                confirmed_by_user = excluded.confirmed_by_user,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            (
                question_normalized,
                question_sample,
                answer,
                scope,
                company_id,
                int(confirmed_by_user),
            ),
        )
        await self.conn.commit()

    async def lookup_qa_cache(
        self,
        question_normalized: str,
        *,
        company_id: int | None = None,
    ) -> str | None:
        """Return confirmed cached answer (company-specific first, then global)."""
        if company_id is not None:
            cursor = await self.conn.execute(
                """
                SELECT answer FROM qa_cache
                WHERE question_normalized = ? AND scope = 'company'
                  AND company_id = ? AND confirmed_by_user = 1
                """,
                (question_normalized, company_id),
            )
            row = await cursor.fetchone()
            if row is not None:
                return str(row["answer"])

        cursor = await self.conn.execute(
            """
            SELECT answer FROM qa_cache
            WHERE question_normalized = ? AND scope = 'global' AND confirmed_by_user = 1
            """,
            (question_normalized,),
        )
        row = await cursor.fetchone()
        return str(row["answer"]) if row is not None else None

    async def start_run(self, run_type: RunType) -> int:
        """Record the start of a pipeline execution."""
        cursor = await self.conn.execute(
            "INSERT INTO runs (run_type, status) VALUES (?, ?)",
            (run_type.value, RunStatus.RUNNING.value),
        )
        await self.conn.commit()
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    async def finish_run(
        self,
        run_id: int,
        *,
        status: RunStatus,
        stats: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finalize a pipeline run record."""
        await self.conn.execute(
            """
            UPDATE runs
            SET status = ?,
                finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                stats_json = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                status.value,
                json.dumps(stats) if stats is not None else None,
                error_message,
                run_id,
            ),
        )
        await self.conn.commit()

    async def get_funnel_stats(self) -> dict[str, int]:
        """Return job counts grouped by lifecycle status."""
        counts: dict[str, int] = {status.value: 0 for status in JobStatus}
        cursor = await self.conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM jobs GROUP BY status"
        )
        rows = await cursor.fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["cnt"])
        return counts
