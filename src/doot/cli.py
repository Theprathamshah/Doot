"""Typer CLI entrypoints for Doot."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from doot.config import DootSettings
from doot.db import DootRepository, InvalidStateTransitionError, initialize_schema, open_connection
from doot.kill_switch import assert_not_stopped, get_safety_status
from doot.logging import configure_logging
from doot.models.enums import JobStatus

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
    assert_not_stopped(settings)

    async def _transition() -> None:
        async with open_connection(settings.db_path) as conn:
            await initialize_schema(conn)
            repo = DootRepository(conn)
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
