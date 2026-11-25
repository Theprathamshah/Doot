"""Typer CLI entrypoints for Doot."""

import asyncio
import webbrowser
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any, TypeVar

import httpx
import typer
from rich.console import Console
from rich.table import Table

from doot.collectors import DomainRateLimiter
from doot.collectors.dork import build_search_provider
from doot.config import DootSettings
from doot.db import DootRepository, InvalidStateTransitionError, initialize_schema, open_connection
from doot.kill_switch import SafetyGuardError, assert_not_stopped, get_safety_status
from doot.logging import configure_logging
from doot.models.enums import ATSProvider, JobStatus, RunStatus, RunType
from doot.notify import dispatch_one_way_alert, render_digest_table, render_job_panel
from doot.pipeline import (
    add_company_from_target,
    load_resume_markdown,
    load_user_profile,
    run_discovery_pipeline,
    run_scoring_pipeline,
    sync_watchlist_from_yaml,
)

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
    dork: bool = typer.Option(False, "--dork/--no-dork", help="Run search-dork board discovery."),
    jobspy: bool = typer.Option(False, "--jobspy/--no-jobspy", help="Run logged-out JobSpy."),
    limit_companies: int | None = typer.Option(None, "--limit", help="Max ATS boards to poll."),
) -> None:
    """Run hybrid job discovery (Dork-to-Crawl + ATS JSON APIs + optional JobSpy)."""
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
            if dork and provider is None:
                console.print(
                    "[yellow]Warning:[/yellow] --dork enabled but no API key configured "
                    f"for {settings.search_provider}; skipping dorks."
                )
            return await run_discovery_pipeline(
                settings,
                repo,
                client=client,
                rate_limiter=rate_limiter,
                use_ats=ats,
                use_dork=dork and provider is not None,
                use_jobspy=jobspy or settings.jobspy_enabled,
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
) -> None:
    """Run Stage 1 deterministic filtering and Stage 2 cached Gemini scoring."""
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
    table.add_row("Gemini API Calls", str(summary.llm_calls))
    table.add_row("Failed", str(summary.failed))
    console.print(table)


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
    """Interactively triage ranked jobs ([s]kip, [a]pplied, [o]pen, [n]ext, [q]uit)."""
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
                    "Action ([s]kip / [a]pplied / [o]pen / [n]ext / [q]uit)",
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
            console.print("[yellow]Invalid choice. Use s, a, o, n, or q.[/yellow]")


@app.command("pipeline-run")
def pipeline_run(
    ats: Annotated[bool, typer.Option("--ats/--no-ats")] = True,
    dork: Annotated[bool, typer.Option("--dork/--no-dork")] = False,
    jobspy: Annotated[bool, typer.Option("--jobspy/--no-jobspy")] = False,
    notify: Annotated[bool, typer.Option("--notify/--no-notify")] = True,
) -> None:
    """Run discover -> score -> digest end-to-end (for macOS LaunchAgent schedules)."""
    discover(ats=ats, dork=dork, jobspy=jobspy, limit_companies=None)
    settings = _load_settings()
    if settings.profile_path.exists():
        score(limit=200, stage1_only=False)
    else:
        console.print("[yellow]Skipping score: data/profile.json not found.[/yellow]")
    digest(min_score=None, limit=25, notify=notify, mark_notified=True)
