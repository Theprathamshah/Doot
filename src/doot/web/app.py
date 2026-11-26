"""Local Starlette web dashboard & JSON API for browsing and applying to jobs."""

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from doot.collectors import DomainRateLimiter
from doot.collectors.dork import build_search_provider
from doot.config import DootSettings
from doot.db import (
    DootRepository,
    InvalidStateTransitionError,
    initialize_schema,
    open_connection,
)
from doot.kill_switch import SafetyGuardError, assert_not_stopped
from doot.llm import LLMProvider, build_llm_provider
from doot.models.enums import JobStatus
from doot.outreach import find_contacts_for_company
from doot.pipeline import load_resume_markdown, load_user_profile
from doot.rag import load_stories_markdown, tailor_for_job

_DASHBOARD_HTML_PATH = Path(__file__).with_name("dashboard.html")


def _safe_http_url(url: str) -> str:
    parsed = urlparse(url.strip())
    return url.strip() if parsed.scheme in ("http", "https") else ""


def _portal_label(job: Any) -> str:
    if job.ats_provider:
        return str(job.ats_provider.value)
    if job.source and job.source != "dork":
        return str(job.source)
    host = (urlparse(job.apply_url).hostname or "").lower().removeprefix("www.")
    return host or str(job.source)


def _serialize_job(job: Any) -> dict[str, Any]:
    return {
        "id": job.id,
        "company_name": job.company_name,
        "title": job.title,
        "location_raw": job.location_raw,
        "is_remote": job.is_remote,
        "india_eligible": job.india_eligible,
        "alert_only": job.alert_only,
        "ats_provider": job.ats_provider.value if job.ats_provider else None,
        "source": job.source,
        "portal": _portal_label(job),
        "salary_raw": job.salary_raw,
        "apply_url": _safe_http_url(job.apply_url),
        "canonical_url": _safe_http_url(job.canonical_url),
        "status": job.status.value,
        "llm_score": job.llm_score,
        "llm_reasoning": job.llm_reasoning,
        "missing_skills": job.missing_skills,
        "rule_filter_reason": job.rule_filter_reason,
        "discovered_at": job.discovered_at,
    }


def _parse_optional_bool(raw: str | None) -> bool | None:
    val = (raw or "").strip().lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return None


def create_web_app(
    settings: DootSettings | None = None,
    *,
    llm_provider: LLMProvider | None = None,
) -> Starlette:
    """Create the local Starlette dashboard application."""
    cfg = settings or DootSettings()

    async def index(_request: Request) -> HTMLResponse:
        html_content = _DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(html_content)

    async def api_stats(_request: Request) -> JSONResponse:
        async with open_connection(cfg.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            funnel = await repo.get_funnel_stats()
            selected_jobs, selected_count = await repo.query_jobs(view="selected", limit=1)
            _ = selected_jobs
            contacts = await repo.list_contacts(limit=500)
            companies = await repo.list_companies(active_only=True)
            portals = await repo.list_portals()
            return JSONResponse(
                {
                    "funnel": funnel,
                    "selected_count": selected_count,
                    "contacts_count": len(contacts),
                    "active_companies": len(companies),
                    "total_jobs": sum(funnel.values()),
                    "portals": portals,
                }
            )

    async def api_list_jobs(request: Request) -> JSONResponse:
        qp = request.query_params
        view = qp.get("view", "selected")
        q = qp.get("q", "")
        source = qp.get("source", "")
        portal = qp.get("portal", "")
        ats_only = qp.get("ats_only", "false").lower() == "true"
        remote_only = _parse_optional_bool(qp.get("remote"))
        india_only = qp.get("india_only", "false").lower() == "true"
        has_salary = qp.get("has_salary", "false").lower() == "true"
        unscored_only = qp.get("unscored_only", "false").lower() == "true"
        status_filter = qp.get("status", "")
        sort_by = qp.get("sort_by", "score_best")
        sort_dir = qp.get("sort_dir", "desc")
        added_days_raw = qp.get("added_days", "").strip()
        try:
            min_score = int(qp.get("min_score", "0"))
            added_within_days = int(added_days_raw) if added_days_raw else None
            limit = min(max(int(qp.get("limit", "100")), 1), 500)
            offset = max(int(qp.get("offset", "0")), 0)
        except ValueError:
            return JSONResponse({"error": "Invalid numeric query parameter"}, status_code=400)

        async with open_connection(cfg.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            jobs, total = await repo.query_jobs(
                view=view,
                query_text=q,
                min_score=min_score,
                unscored_only=unscored_only,
                source=source,
                portal=portal,
                ats_only=ats_only,
                remote_only=remote_only,
                india_only=india_only,
                has_salary=has_salary,
                added_within_days=added_within_days,
                status_filter=status_filter,
                sort_by=sort_by,
                sort_dir=sort_dir,
                limit=limit,
                offset=offset,
            )
            return JSONResponse(
                {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "jobs": [_serialize_job(j) for j in jobs],
                }
            )

    async def api_job_detail(request: Request) -> JSONResponse:
        job_id = int(request.path_params["job_id"])
        async with open_connection(cfg.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            job = await repo.get_job(job_id)
            if job is None:
                return JSONResponse({"error": f"Job #{job_id} not found"}, status_code=404)
            application = await repo.get_application(job_id)
            contacts = await repo.list_contacts(job_id=job_id, limit=20)
            payload = _serialize_job(job)
            payload["jd_text"] = job.jd_text
            payload["application"] = application
            payload["contacts"] = contacts
            return JSONResponse(payload)

    async def api_update_status(request: Request) -> JSONResponse:
        job_id = int(request.path_params["job_id"])
        try:
            assert_not_stopped(cfg)
            body = await request.json()
            target = JobStatus(str(body.get("status", "")).strip().upper())
            reason = str(body.get("reason") or "updated via web UI")
        except (SafetyGuardError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        async with open_connection(cfg.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            try:
                if target == JobStatus.APPLIED:
                    current = await repo.get_job(job_id)
                    if current is not None and current.status == JobStatus.SCORED:
                        await repo.transition_job(
                            job_id,
                            JobStatus.NOTIFIED,
                            actor="web_ui",
                            reason="surfaced in web UI before applied",
                        )
                updated = await repo.transition_job(job_id, target, actor="web_ui", reason=reason)
                return JSONResponse({"id": updated.id, "status": updated.status.value})
            except LookupError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except InvalidStateTransitionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)

    async def api_tailor_job(request: Request) -> JSONResponse:
        job_id = int(request.path_params["job_id"])
        try:
            assert_not_stopped(cfg)
            profile = load_user_profile(cfg)
            provider = llm_provider or build_llm_provider(cfg)
        except (SafetyGuardError, FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        resume_md = load_resume_markdown(profile, cfg)
        stories_md = load_stories_markdown(profile, cfg)

        async with open_connection(cfg.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            try:
                assets = await tailor_for_job(
                    repo,
                    provider,
                    job_id=job_id,
                    profile=profile,
                    resume_markdown=resume_md,
                    stories_markdown=stories_md,
                )
                return JSONResponse(assets.model_dump(mode="json"))
            except LookupError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=500)

    async def api_find_contacts(request: Request) -> JSONResponse:
        job_id = int(request.path_params["job_id"])
        try:
            assert_not_stopped(cfg)
        except SafetyGuardError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        async with (
            open_connection(cfg.db_path) as conn,
            httpx.AsyncClient() as client,
        ):
            await initialize_schema(conn)
            repo = DootRepository(conn)
            job = await repo.get_job(job_id)
            if job is None:
                return JSONResponse({"error": f"Job #{job_id} not found"}, status_code=404)
            careers_url: str | None = None
            if job.company_id is not None:
                co = await repo.get_company_by_id(job.company_id)
                if co is not None:
                    careers_url = co.careers_url
            if not careers_url:
                careers_url = job.apply_url

            rate_limiter = DomainRateLimiter(min_interval_seconds=1.0)
            search_provider = build_search_provider(cfg, client, rate_limiter)
            res = await find_contacts_for_company(
                cfg,
                repo,
                company_name=job.company_name,
                careers_url=careers_url,
                jd_text=job.jd_text,
                company_id=job.company_id,
                job_id=job.id,
                search_provider=search_provider,
                client=client,
            )
            contacts = await repo.list_contacts(job_id=job_id, limit=20)
            return JSONResponse(
                {
                    "domain": res.domain,
                    "mx_valid": res.mx_valid,
                    "contacts": contacts,
                }
            )

    async def api_delete_job(request: Request) -> JSONResponse:
        job_id = int(request.path_params["job_id"])
        try:
            assert_not_stopped(cfg)
        except SafetyGuardError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        async with open_connection(cfg.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            deleted = await repo.delete_job(job_id)
            if not deleted:
                return JSONResponse({"error": f"Job #{job_id} not found"}, status_code=404)
            return JSONResponse({"deleted": True, "id": job_id})

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/stats", api_stats, methods=["GET"]),
        Route("/api/jobs", api_list_jobs, methods=["GET"]),
        Route("/api/jobs/{job_id:int}", api_job_detail, methods=["GET"]),
        Route("/api/jobs/{job_id:int}", api_delete_job, methods=["DELETE"]),
        Route("/api/jobs/{job_id:int}/status", api_update_status, methods=["POST"]),
        Route("/api/jobs/{job_id:int}/tailor", api_tailor_job, methods=["POST"]),
        Route("/api/jobs/{job_id:int}/find-contacts", api_find_contacts, methods=["POST"]),
    ]
    return Starlette(routes=routes)
