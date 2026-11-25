"""Doot MCP Server exposing job triage, RAG tailoring, and funnel tools over stdio."""

from typing import Any

from mcp.server.mcpserver import MCPServer

from doot.config import DootSettings
from doot.db import DootRepository, initialize_schema, open_connection
from doot.kill_switch import assert_not_stopped
from doot.llm import LLMProvider, build_llm_provider
from doot.models.enums import JobStatus
from doot.pipeline.scorer import load_resume_markdown, load_user_profile
from doot.rag import load_stories_markdown, tailor_for_job


async def mcp_list_top_jobs(
    settings: DootSettings,
    *,
    min_score: int = 4,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return top-ranked scored/notified jobs from SQLite."""
    async with open_connection(settings.db_path) as conn:
        await initialize_schema(conn)
        repo = DootRepository(conn)
        jobs = await repo.list_jobs_by_statuses(
            (JobStatus.SCORED, JobStatus.NOTIFIED, JobStatus.PREPARED),
            min_score=min_score,
            limit=limit,
        )
        return [
            {
                "id": j.id,
                "company": j.company_name,
                "title": j.title,
                "location": j.location_raw,
                "score": j.llm_score,
                "status": j.status.value,
                "alert_only": j.alert_only,
                "apply_url": j.apply_url,
                "reasoning": j.llm_reasoning,
                "missing_skills": j.missing_skills,
            }
            for j in jobs
        ]


async def mcp_get_job_details(settings: DootSettings, job_id: int) -> dict[str, Any]:
    """Fetch full job details, stored tailoring, and audit events."""
    async with open_connection(settings.db_path) as conn:
        await initialize_schema(conn)
        repo = DootRepository(conn)
        job = await repo.get_job(job_id)
        if job is None:
            raise LookupError(f"Job #{job_id} not found")
        application = await repo.get_application(job_id)
        events = await repo.get_job_events(job_id)
        return {
            "job": job.model_dump(mode="json"),
            "application": application,
            "events": events,
        }


async def mcp_tailor_for_job(
    settings: DootSettings,
    job_id: int,
    *,
    llm_provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Run RAG-grounded tailoring for a job and store results."""
    assert_not_stopped(settings)
    profile = load_user_profile(settings)
    resume_md = load_resume_markdown(profile, settings)
    stories_md = load_stories_markdown(profile, settings)
    provider = llm_provider or build_llm_provider(settings)

    async with open_connection(settings.db_path) as conn:
        await initialize_schema(conn)
        repo = DootRepository(conn)
        assets = await tailor_for_job(
            repo,
            provider,
            job_id=job_id,
            profile=profile,
            resume_markdown=resume_md,
            stories_markdown=stories_md,
        )
        return assets.model_dump(mode="json")


async def mcp_mark_job_status(
    settings: DootSettings,
    job_id: int,
    status: str,
    *,
    reason: str = "updated via MCP",
) -> dict[str, Any]:
    """Validate and apply a job lifecycle transition."""
    assert_not_stopped(settings)
    target = JobStatus(status.strip().upper())

    async with open_connection(settings.db_path) as conn:
        await initialize_schema(conn)
        repo = DootRepository(conn)
        if target == JobStatus.APPLIED:
            current = await repo.get_job(job_id)
            if current is not None and current.status == JobStatus.SCORED:
                await repo.transition_job(
                    job_id,
                    JobStatus.NOTIFIED,
                    actor="mcp",
                    reason="surfaced via MCP before applied",
                )
        updated = await repo.transition_job(job_id, target, actor="mcp", reason=reason)
        return {"id": updated.id, "status": updated.status.value}


async def mcp_get_funnel_stats(settings: DootSettings) -> dict[str, int]:
    """Return job counts grouped by lifecycle status."""
    async with open_connection(settings.db_path) as conn:
        await initialize_schema(conn)
        repo = DootRepository(conn)
        return await repo.get_funnel_stats()


def create_mcp_server(settings: DootSettings | None = None) -> MCPServer:
    """Create and configure the Doot MCPServer instance."""
    cfg = settings or DootSettings()
    mcp = MCPServer("doot")

    @mcp.tool()
    async def list_top_jobs(min_score: int = 4, limit: int = 20) -> list[dict[str, Any]]:
        """List top-scored actionable jobs from the local Doot database."""
        return await mcp_list_top_jobs(cfg, min_score=min_score, limit=limit)

    @mcp.tool()
    async def get_job_details(job_id: int) -> dict[str, Any]:
        """Get full JD, score reasoning, stored tailoring, and events for a job."""
        return await mcp_get_job_details(cfg, job_id)

    @mcp.tool()
    async def tailor_job(job_id: int) -> dict[str, Any]:
        """Generate RAG-grounded tailored summary, cover note, and outreach DM for a job."""
        return await mcp_tailor_for_job(cfg, job_id)

    @mcp.tool()
    async def mark_job_status(
        job_id: int,
        status: str,
        reason: str = "updated via MCP",
    ) -> dict[str, Any]:
        """Transition a job status (e.g. SKIPPED, APPLIED) enforcing state rules."""
        return await mcp_mark_job_status(cfg, job_id, status, reason=reason)

    @mcp.tool()
    async def get_funnel_stats() -> dict[str, int]:
        """Return job funnel counts grouped by lifecycle state."""
        return await mcp_get_funnel_stats(cfg)

    return mcp
