"""Typer CLI entrypoints for Doot."""

import asyncio
import webbrowser
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any, TypeVar

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from doot.applier import FieldResolver, FilledFormResult, load_aliases_config, run_assisted_apply
from doot.collectors import DomainRateLimiter
from doot.collectors.dork import build_search_provider
from doot.config import DootSettings
from doot.db import DootRepository, InvalidStateTransitionError, initialize_schema, open_connection
from doot.kill_switch import SafetyGuardError, assert_not_stopped, get_safety_status
from doot.llm import build_llm_provider
from doot.logging import configure_logging
from doot.mcp_server import create_mcp_server
from doot.models.enums import ATSProvider, JobStatus, RunStatus, RunType
from doot.models.job import TailoredAssets
from doot.notify import dispatch_one_way_alert, render_digest_table, render_job_panel
from doot.outreach import (
    ColdEmailDraft,
    draft_cold_email,
    find_contacts_for_company,
    send_outreach_email,
)
from doot.pipeline import (
    add_company_from_target,
    load_resume_markdown,
    load_user_profile,
    run_discovery_pipeline,
    run_scoring_pipeline,
    sync_watchlist_from_yaml,
)
from doot.rag import collect_evidence_chunks, load_stories_markdown, tailor_for_job
from doot.web import create_web_app

T = TypeVar("T")

app = typer.Typer(
    name="doot",
    help="Local-first Job Search Triage, RAG Tailoring, MCP & Assisted-Apply agent.",
    no_args_is_help=True,
)
console = Console()


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a synchronous Typer command."""
    return asyncio.run(coro)


def _load_settings() -> DootSettings:
    settings = DootSettings()
    configure_logging(settings)
    return settings


@app.command()
def init() -> None:
    """Initialize data directory and SQLite WAL database schema."""
    settings = _load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    async def _init() -> None:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)

    _run_async(_init())
    console.print(f"[green]Initialized database:[/green] {settings.db_path}")
    if not settings.profile_path.exists():
        console.print(
            f"[yellow]Note:[/yellow] Copy data/profile.example.json to "
            f"{settings.profile_path} before Phase 3."
        )


@app.command()
def status() -> None:
    """Show configuration, kill-switch, and safety guardrail status."""
    settings = _load_settings()

    async def _status() -> tuple[str, int, int, bool]:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            cursor = await conn.execute("PRAGMA journal_mode;")
            mode_row = await cursor.fetchone()
            journal_mode = str(mode_row[0]) if mode_row is not None else "unknown"
            repo = DootRepository(conn)
            safety = await get_safety_status(settings, repo)
            return journal_mode, safety.applies_today, safety.daily_cap, safety.stop_file_active

    journal_mode, applies_today, daily_cap, stop_active = _run_async(_status())

    table = Table(title="Doot Status")
    table.add_column("Check", style="cyan")
    table.add_column("Value")
    stop_label = "[red]ACTIVE[/red]" if stop_active else "[green]OFF[/green]"
    key_label = (
        "[green]configured[/green]"
        if settings.gemini_api_key.get_secret_value()
        else "[yellow]missing[/yellow]"
    )
    table.add_row("Database", f"{settings.db_path} (journal_mode={journal_mode})")
    table.add_row("Kill Switch (data/STOP)", stop_label)
    table.add_row("Dry-Run Mode", str(settings.dry_run))
    table.add_row("Daily Automated Applies", f"{applies_today} / {daily_cap}")
    table.add_row("LLM Provider / Model", f"{settings.llm_provider} ({settings.gemini_model})")
    table.add_row("Gemini API Key", key_label)
    table.add_row(
        "Profile JSON",
        "[green]found[/green]" if settings.profile_path.exists() else "[yellow]missing[/yellow]",
    )
    console.print(table)


@app.command()
def stats() -> None:
    """Display job funnel counts by lifecycle state."""
    settings = _load_settings()

    async def _stats() -> dict[str, int]:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await repo.get_funnel_stats()

    counts = _run_async(_stats())
    table = Table(title="Job Funnel Stats")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    total = 0
    for state, count in counts.items():
        table.add_row(state, str(count))
        total += count
    table.add_row("TOTAL", str(total), style="bold")
    console.print(table)


def _transition_command(job_id: int, target: JobStatus, reason: str) -> None:
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
    except SafetyGuardError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    async def _transition() -> None:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            if target == JobStatus.APPLIED:
                current = await repo.get_job(job_id)
                if current is not None and current.status == JobStatus.SCORED:
                    await repo.transition_job(
                        job_id,
                        JobStatus.NOTIFIED,
                        actor="cli",
                        reason="surfaced before marking applied",
                    )
            await repo.transition_job(job_id, target, actor="cli", reason=reason)

    try:
        _run_async(_transition())
    except (LookupError, InvalidStateTransitionError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Job {job_id} marked {target.value}[/green]")


@app.command()
def skip(job_id: int, reason: str = typer.Option("skipped via CLI", "--reason", "-r")) -> None:
    """Mark a job as SKIPPED."""
    _transition_command(job_id, JobStatus.SKIPPED, reason)


@app.command()
def applied(
    job_id: int,
    reason: str = typer.Option("marked applied via CLI", "--reason", "-r"),
) -> None:
    """Mark a job as APPLIED (for manual or alert-only applications)."""
    _transition_command(job_id, JobStatus.APPLIED, reason)


@app.command()
def stop() -> None:
    """Activate the global kill switch by creating data/STOP."""
    settings = _load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.stop_file_path.write_text("stopped\n", encoding="utf-8")
    console.print(f"[red]Kill switch activated:[/red] {settings.stop_file_path}")


@app.command()
def resume() -> None:
    """Deactivate the global kill switch by removing data/STOP."""
    settings = _load_settings()
    if settings.stop_file_path.exists():
        settings.stop_file_path.unlink()
        console.print("[green]Kill switch removed.[/green]")
    else:
        console.print("Kill switch is not active.")


@app.command("sync-watchlist")
def sync_watchlist(
    path: Annotated[
        Path | None,
        typer.Argument(help="Path to companies YAML watchlist."),
    ] = None,
) -> None:
    """Load or refresh companies from data/companies.yaml into SQLite."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
    except SafetyGuardError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    target_path = path or settings.companies_path

    async def _sync() -> int:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await sync_watchlist_from_yaml(repo, target_path)

    try:
        count = _run_async(_sync())
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Synced {count} companies from {target_path}[/green]")


@app.command("add-company")
def add_company(
    target: Annotated[
        str,
        typer.Argument(help="Careers URL, ATS board URL, or label."),
    ],
    ats: Annotated[
        ATSProvider | None,
        typer.Option("--ats", help="Explicit ATS provider."),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Explicit ATS board token."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Display company name."),
    ] = None,
) -> None:
    """Auto-detect Greenhouse/Lever/Ashby board from a URL and add to watchlist."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
    except SafetyGuardError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    async def _add() -> tuple[int, ATSProvider, str]:
        async with (
            open_connection(settings.db_path) as conn,
            httpx.AsyncClient() as client,
        ):
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await add_company_from_target(
                repo,
                target,
                client=client,
                name=name,
                ats_provider=ats,
                board_token=token,
            )

    try:
        company_id, provider, board_token = _run_async(_add())
    except (ValueError, httpx.HTTPError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Added company #{company_id}:[/green] {provider.value}:{board_token}"
    )


@app.command()
def discover(
    ats: bool = typer.Option(True, "--ats/--no-ats", help="Poll active ATS JSON boards."),
    feeds: bool = typer.Option(
        True,
        "--feeds/--no-feeds",
        help="Poll free public remote/startup JSON feeds (Himalayas, Remotive, Jobicy, HN).",
    ),
    dork: bool = typer.Option(False, "--dork/--no-dork", help="Run search-dork board discovery."),
    dork_priority: Annotated[
        int,
        typer.Option(
            "--dork-priority",
            help="Max dork priority level to run (1=ATS only, 2=+boards, 3=+social).",
        ),
    ] = 2,
    jobspy: bool = typer.Option(False, "--jobspy/--no-jobspy", help="Run logged-out JobSpy."),
    limit_companies: int | None = typer.Option(None, "--limit", help="Max ATS boards to poll."),
) -> None:
    """Run hybrid job discovery (Dork-to-Crawl + ATS JSON + Public Feeds + optional JobSpy)."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
    except SafetyGuardError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    async def _discover() -> Any:
        rate_limiter = DomainRateLimiter(min_interval_seconds=settings.ats_min_interval_seconds)
        async with (
            open_connection(settings.db_path) as conn,
            httpx.AsyncClient() as client,
        ):
            await initialize_schema(conn)
            repo = DootRepository(conn)
            provider = build_search_provider(settings, client, rate_limiter) if dork else None
            return await run_discovery_pipeline(
                settings,
                repo,
                client=client,
                rate_limiter=rate_limiter,
                use_ats=ats,
                use_feeds=feeds,
                use_dork=dork and provider is not None,
                use_jobspy=jobspy or settings.jobspy_enabled,
                dork_max_priority=dork_priority,
                limit_companies=limit_companies,
                search_provider=provider,
            )

    summary = _run_async(_discover())
    table = Table(title="Discovery Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("ATS Boards Crawled", str(summary.companies_crawled))
    table.add_row("ATS Boards Failed", str(summary.companies_failed))
    table.add_row("ATS Boards Auto-Paused", str(summary.companies_paused))
    table.add_row("New Boards via Dork", str(summary.new_boards_via_dork))
    table.add_row("Jobs Created", str(summary.jobs_created))
    table.add_row("Jobs Updated", str(summary.jobs_updated))
    table.add_row("Cross-Source Deduped", str(summary.jobs_deduped_cross_source))
    console.print(table)


@app.command()
def score(
    limit: Annotated[
        int,
        typer.Option("--limit", help="Max DISCOVERED jobs to evaluate."),
    ] = 200,
    stage1_only: Annotated[
        bool,
        typer.Option("--stage1-only", help="Run only Stage 1 rule filter (zero LLM calls)."),
    ] = False,
    retry_failed: Annotated[
        bool,
        typer.Option("--retry-failed", help="Also re-evaluate previously FAILED jobs."),
    ] = False,
    local: Annotated[
        bool,
        typer.Option(
            "--local",
            help="Use instant zero-cost local heuristic scorer instead of Gemini API.",
        ),
    ] = False,
) -> None:
    """Run Stage 1 deterministic filtering and Stage 2 scoring (Gemini or --local)."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
        profile = load_user_profile(settings)
    except (SafetyGuardError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    resume_md = load_resume_markdown(profile, settings)

    async def _score() -> Any:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await run_scoring_pipeline(
                settings,
                repo,
                profile=profile,
                resume_markdown=resume_md,
                limit=limit,
                stage1_only=stage1_only,
                retry_failed=retry_failed,
                local_only=local,
            )

    try:
        summary = _run_async(_score())
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Scoring Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Evaluated", str(summary.evaluated))
    table.add_row("Filtered Out (Stage 1)", str(summary.filtered_out))
    table.add_row("Scored (Stage 2)", str(summary.scored))
    table.add_row("Cache Hits", str(summary.cache_hits))
    table.add_row("Scorer Calls", str(summary.llm_calls))
    table.add_row("Failed", str(summary.failed))
    console.print(table)
    if summary.quota_exhausted:
        console.print(
            "[yellow]Gemini daily free-tier quota exhausted.[/yellow] "
            "Run `uv run doot score --local --retry-failed` for instant free scoring."
        )


@app.command()
def digest(
    min_score: Annotated[
        int | None,
        typer.Option("--min-score", help="Minimum LLM score (defaults to DOOT_MIN_DIGEST_SCORE)."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum jobs to display."),
    ] = 25,
    notify: Annotated[
        bool,
        typer.Option("--notify/--no-notify", help="Send one-way alert using DOOT_NOTIFY_METHOD."),
    ] = False,
    mark_notified: Annotated[
        bool,
        typer.Option("--mark-notified/--no-mark-notified", help="Transition SCORED -> NOTIFIED."),
    ] = True,
) -> None:
    """Display ranked matching jobs and optionally send a one-way alert."""
    settings = _load_settings()
    threshold = min_score if min_score is not None else settings.min_digest_score

    async def _digest() -> tuple[list[Any], int, bool]:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            run_id = await repo.start_run(RunType.DIGEST)
            try:
                jobs = await repo.list_jobs_by_statuses(
                    (JobStatus.SCORED, JobStatus.NOTIFIED),
                    min_score=threshold,
                    limit=limit,
                )
                sent = False
                if notify and jobs:
                    sent = await dispatch_one_way_alert(settings, jobs)
                transitioned = 0
                if mark_notified and (notify or jobs):
                    transitioned = await repo.mark_jobs_notified([j.id for j in jobs])
                await repo.finish_run(
                    run_id,
                    status=RunStatus.SUCCESS,
                    stats={"jobs": len(jobs), "notified": transitioned, "alert_sent": sent},
                )
                return jobs, transitioned, sent
            except Exception as exc:
                await repo.finish_run(run_id, status=RunStatus.FAILED, error_message=str(exc))
                raise

    jobs, transitioned, sent = _run_async(_digest())
    render_digest_table(jobs, console)
    if transitioned:
        console.print(f"[dim]Transitioned {transitioned} SCORED job(s) to NOTIFIED.[/dim]")
    if notify:
        msg = "Alert sent." if sent else "No alert sent (empty or disabled)."
        console.print(f"[dim]{msg}[/dim]")


def _render_tailored_assets(job_id: int, assets: Any) -> None:
    citations = ", ".join(assets.cited_story_ids) if assets.cited_story_ids else "none"
    console.print(Panel(assets.tailored_summary, title=f"Job #{job_id} — Tailored Summary"))
    console.print(Panel(assets.cover_note, title=f"Job #{job_id} — Cover Note"))
    console.print(Panel(assets.outreach_dm, title=f"Job #{job_id} — Recruiter Outreach DM"))
    console.print(f"[dim]Verified Evidence Citations: {citations}[/dim]")


def _run_tailor(settings: DootSettings, job_id: int) -> Any:
    profile = load_user_profile(settings)
    resume_md = load_resume_markdown(profile, settings)
    stories_md = load_stories_markdown(profile, settings)
    provider = build_llm_provider(settings)

    async def _tailor() -> Any:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await tailor_for_job(
                repo,
                provider,
                job_id=job_id,
                profile=profile,
                resume_markdown=resume_md,
                stories_markdown=stories_md,
            )

    return _run_async(_tailor())


@app.command()
def review(
    min_score: Annotated[
        int | None,
        typer.Option("--min-score", help="Minimum score for review queue."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum jobs in queue."),
    ] = 20,
) -> None:
    """Interactively triage ranked jobs ([t]ailor, [s]kip, [a]pplied, [o]pen, [n]ext, [q]uit)."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
    except SafetyGuardError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    threshold = min_score if min_score is not None else settings.min_digest_score

    async def _load_queue() -> list[Any]:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await repo.list_jobs_by_statuses(
                (JobStatus.SCORED, JobStatus.NOTIFIED, JobStatus.NEEDS_MANUAL_ACTION),
                min_score=threshold,
                limit=limit,
            )

    queue = _run_async(_load_queue())
    if not queue:
        console.print("[yellow]Review queue is empty.[/yellow]")
        return

    for job in queue:
        render_job_panel(job, console)
        while True:
            choice = (
                typer.prompt(
                    "Action ([t]ailor / [s]kip / [a]pplied / [o]pen / [n]ext / [q]uit)",
                    default="n",
                )
                .strip()
                .lower()
            )
            if choice in ("q", "quit"):
                console.print("Exiting review.")
                return
            if choice in ("n", "next"):
                break
            if choice in ("t", "tailor"):
                try:
                    assets = _run_tailor(settings, job.id)
                    _render_tailored_assets(job.id, assets)
                except (FileNotFoundError, LookupError, ValueError) as exc:
                    console.print(f"[red]Tailor error:[/red] {exc}")
                continue
            if choice in ("o", "open"):
                webbrowser.open(job.apply_url)
                console.print(f"[dim]Opened {job.apply_url}[/dim]")
                continue
            if choice in ("s", "skip"):
                _transition_command(job.id, JobStatus.SKIPPED, "skipped during CLI review")
                break
            if choice in ("a", "applied"):
                _transition_command(job.id, JobStatus.APPLIED, "applied during CLI review")
                break
            console.print("[yellow]Invalid choice. Use t, s, a, o, n, or q.[/yellow]")


@app.command()
def tailor(
    job_id: Annotated[int, typer.Argument(help="Database ID of the job to tailor for.")],
) -> None:
    """Generate RAG-grounded summary, cover note, and outreach DM for a job."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
        assets = _run_tailor(settings, job_id)
    except (SafetyGuardError, FileNotFoundError, LookupError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _render_tailored_assets(job_id, assets)


@app.command()
def mcp() -> None:
    """Run Doot's Model Context Protocol (MCP) server over stdio."""
    settings = _load_settings()
    server = create_mcp_server(settings)
    server.run(transport="stdio")


@app.command("apply")
def apply_job(
    job_id: Annotated[int, typer.Argument(help="Database ID of the ATS job to fill.")],
    live_apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Allow form submission after confirmation (default: dry-run).",
        ),
    ] = False,
    headless: Annotated[
        bool,
        typer.Option("--headless", help="Run Chromium headless instead of headed."),
    ] = False,
) -> None:
    """Pre-fill Greenhouse/Lever/Ashby form in headed Chromium and confirm before submit."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
        profile = load_user_profile(settings)
    except (SafetyGuardError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    aliases = load_aliases_config(settings.data_dir / "aliases.yaml")
    resume_md = load_resume_markdown(profile, settings)
    stories_md = load_stories_markdown(profile, settings)
    evidence = [
        c.render_for_prompt()
        for c in collect_evidence_chunks(
            profile, resume_markdown=resume_md, stories_markdown=stories_md
        )
    ]
    llm = (
        build_llm_provider(settings)
        if settings.gemini_api_key.get_secret_value()
        else None
    )

    async def _ask_cli(question: str) -> str | None:
        ans = typer.prompt(f"Unknown ATS question '{question}' (blank to skip)", default="")
        return str(ans) if str(ans).strip() else None

    async def _confirm_submit(filled: FilledFormResult) -> bool:
        table = Table(title=f"Prepared Fields for Job #{job_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for k, v in filled.resolved_fields.items():
            table.add_row(k, v[:120])
        console.print(table)
        if filled.unknown_questions:
            unresolved = ", ".join(filled.unknown_questions)
            console.print(f"[yellow]Unresolved required fields:[/yellow] {unresolved}")
        console.print(f"[dim]Screenshot saved to: {filled.screenshot_path}[/dim]")
        return typer.confirm("Submit this application now?", default=False)

    async def _run() -> Any:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            stored_app = await repo.get_application(job_id)
            assets = (
                TailoredAssets(
                    tailored_summary=stored_app["tailored_summary"] or "",
                    cover_note=stored_app["cover_note"] or "",
                    outreach_dm=stored_app["outreach_note"] or "",
                )
                if stored_app is not None
                else None
            )
            resolver = FieldResolver(
                profile=profile,
                aliases=aliases,
                repo=repo,
                llm_provider=llm,
                evidence_chunks=evidence,
                ask_user_fn=_ask_cli,
            )
            return await run_assisted_apply(
                settings,
                repo,
                job_id=job_id,
                profile=profile,
                field_resolver=resolver,
                tailored_assets=assets,
                apply_flag=live_apply,
                headless=headless,
                confirm_submit_fn=_confirm_submit,
            )

    try:
        result = _run_async(_run())
    except (SafetyGuardError, LookupError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    color = "green" if result.submitted else "yellow"
    console.print(f"[{color}]{result.status.value}:[/{color}] {result.message}")


@app.command("find-contacts")
def find_contacts(
    job_id: Annotated[
        int | None,
        typer.Option("--job-id", help="Lookup HR/manager/founder contacts for a saved job."),
    ] = None,
    company: Annotated[
        str | None,
        typer.Option("--company", help="Target company name (if not using --job-id)."),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option("--domain", help="Explicit company domain (e.g. razorpay.com)."),
    ] = None,
) -> None:
    """Find public HR, Engineering Manager, or Founder emails for a job or company."""
    settings = _load_settings()
    if job_id is None and not company:
        console.print("[red]Error:[/red] Provide either --job-id or --company.")
        raise typer.Exit(code=1)

    async def _find() -> tuple[Any, list[dict[str, Any]]]:
        rate_limiter = DomainRateLimiter(min_interval_seconds=1.0)
        async with (
            open_connection(settings.db_path) as conn,
            httpx.AsyncClient() as client,
        ):
            await initialize_schema(conn)
            repo = DootRepository(conn)
            search_provider = build_search_provider(settings, client, rate_limiter)

            target_company = company or ""
            careers_url: str | None = None
            jd_text = ""
            company_id: int | None = None

            if job_id is not None:
                job = await repo.get_job(job_id)
                if job is None:
                    raise LookupError(f"Job #{job_id} not found")
                target_company = target_company or job.company_name
                company_id = job.company_id
                jd_text = job.jd_text
                if job.company_id is not None:
                    co = await repo.get_company_by_id(job.company_id)
                    if co is not None:
                        careers_url = co.careers_url
                if not careers_url:
                    careers_url = job.apply_url

            res = await find_contacts_for_company(
                settings,
                repo,
                company_name=target_company,
                domain=domain,
                careers_url=careers_url,
                jd_text=jd_text,
                company_id=company_id,
                job_id=job_id,
                search_provider=search_provider,
                client=client,
            )
            saved = await repo.list_contacts(job_id=job_id, limit=30)
            return res, [c for c in saved if c["id"] in res.contact_ids]

    try:
        res, rows = _run_async(_find())
    except LookupError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    mx_label = "[green]valid MX[/green]" if res.mx_valid else "[yellow]unverified MX[/yellow]"
    console.print(f"Company: [bold]{res.company_name}[/bold] | Domain: {res.domain} ({mx_label})")

    table = Table(title=f"Discovered Contacts ({len(rows)})")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Email")
    table.add_column("Confidence")
    table.add_column("Source")
    for row in rows:
        table.add_row(
            f"#{row['id']}",
            str(row["person_name"] or "-"),
            str(row["role_title"] or "-"),
            str(row["email"]),
            str(row["confidence"]),
            str(row["source"]),
        )
    console.print(table)


@app.command()
def contacts(
    job_id: Annotated[int | None, typer.Option("--job-id")] = None,
    unemailed: Annotated[bool, typer.Option("--unemailed")] = False,
    limit: Annotated[int, typer.Option("--limit")] = 50,
) -> None:
    """List saved HR/manager/founder contacts."""
    settings = _load_settings()

    async def _list() -> list[dict[str, Any]]:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            return await repo.list_contacts(
                job_id=job_id, unemailed_only=unemailed, limit=limit
            )

    rows = _run_async(_list())
    if not rows:
        console.print("[yellow]No contacts found.[/yellow]")
        return

    table = Table(title=f"Saved Contacts ({len(rows)})")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Company")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Email")
    table.add_column("Confidence")
    table.add_column("Emailed")
    for row in rows:
        table.add_row(
            f"#{row['id']}",
            str(row["company_name"]),
            str(row["person_name"] or "-"),
            str(row["role_title"] or "-"),
            str(row["email"]),
            str(row["confidence"]),
            str(row["emailed_at"] or "no"),
        )
    console.print(table)


@app.command()
def outreach(
    contact_id: Annotated[int, typer.Argument(help="Contact ID from `doot contacts`.")],
    send: Annotated[
        bool,
        typer.Option("--send", help="Actually send email via SMTP (default: dry-run draft)."),
    ] = False,
    attach_resume: Annotated[
        bool,
        typer.Option("--attach/--no-attach", help="Attach data/resume.pdf when sending."),
    ] = True,
) -> None:
    """Draft or send a grounded cold email to an HR/manager/founder contact."""
    settings = _load_settings()
    try:
        assert_not_stopped(settings)
        profile = load_user_profile(settings)
    except (SafetyGuardError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    resume_md = load_resume_markdown(profile, settings)
    stories_md = load_stories_markdown(profile, settings)
    llm = build_llm_provider(settings) if settings.gemini_api_key.get_secret_value() else None

    async def _confirm(draft: ColdEmailDraft, recipient: str) -> bool:
        return typer.confirm(f"Send cold email to {recipient}?", default=False)

    async def _run() -> tuple[ColdEmailDraft, bool, str]:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
            contact = await repo.get_contact(contact_id)
            if contact is None:
                raise LookupError(f"Contact #{contact_id} not found")
            linked_job = (
                await repo.get_job(int(contact["job_id"]))
                if contact.get("job_id") is not None
                else None
            )
            role_title = linked_job.title if linked_job else "Backend / Full Stack Engineer"
            draft = await draft_cold_email(
                profile=profile,
                company_name=str(contact["company_name"]),
                role_title=role_title,
                person_name=str(contact["person_name"] or ""),
                recipient_role=str(contact["role_title"] or ""),
                job=linked_job,
                resume_markdown=resume_md,
                stories_markdown=stories_md,
                llm_provider=llm,
            )
            console.print(
                Panel(
                    f"[bold]To:[/bold] {contact['email']}\n"
                    f"[bold]Subject:[/bold] {draft.subject}\n\n{draft.body}",
                    title=f"Outreach Draft (Contact #{contact_id})",
                    border_style="cyan",
                )
            )
            attachment = profile.resume_pdf_path if attach_resume else None
            sent = await send_outreach_email(
                settings,
                repo,
                contact_id=contact_id,
                draft=draft,
                job_id=linked_job.id if linked_job else None,
                send_flag=send,
                attachment_path=attachment,
                confirm_fn=_confirm,
            )
            return draft, sent, str(contact["email"])

    try:
        _, sent, recipient = _run_async(_run())
    except (SafetyGuardError, LookupError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if sent:
        console.print(f"[green]Sent cold email to {recipient}.[/green]")
    else:
        console.print(
            "[yellow]Draft saved (dry-run). Pass `--send` with DOOT_DRY_RUN=false "
            "and SMTP configured to send.[/yellow]"
        )


@app.command()
def ui(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8765,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open dashboard URL in default browser."),
    ] = True,
) -> None:
    """Launch the local web dashboard to view all jobs, selected roles, and apply directly."""
    settings = _load_settings()
    url = f"http://{host}:{port}"
    console.print(f"[green]Starting Doot Web UI at[/green] {url}")
    if open_browser:
        webbrowser.open(url)
    web_app = create_web_app(settings)
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


@app.command("pipeline-run")
def pipeline_run(
    ats: Annotated[bool, typer.Option("--ats/--no-ats")] = True,
    feeds: Annotated[bool, typer.Option("--feeds/--no-feeds")] = True,
    dork: Annotated[bool, typer.Option("--dork/--no-dork")] = False,
    dork_priority: Annotated[int, typer.Option("--dork-priority")] = 2,
    jobspy: Annotated[bool, typer.Option("--jobspy/--no-jobspy")] = False,
    notify: Annotated[bool, typer.Option("--notify/--no-notify")] = True,
) -> None:
    """Run discover -> score -> digest end-to-end (for macOS LaunchAgent schedules)."""
    discover(
        ats=ats,
        feeds=feeds,
        dork=dork,
        dork_priority=dork_priority,
        jobspy=jobspy,
        limit_companies=None,
    )
    settings = _load_settings()
    if settings.profile_path.exists():
        score(limit=200, stage1_only=False)
    else:
        console.print("[yellow]Skipping score: data/profile.json not found.[/yellow]")
    digest(min_score=None, limit=25, notify=notify, mark_notified=True)
