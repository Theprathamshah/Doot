"""Digest formatting and optional one-way notification delivery."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from doot.models.job import JobRecord


def build_digest_text(jobs: list[JobRecord]) -> str:
    """Build a compact plain-text ranked digest suitable for alerts or logs."""
    if not jobs:
        return "Doot Digest: No matching jobs above threshold."

    lines = [f"Doot Digest ({len(jobs)} matching role(s)):", ""]
    for idx, job in enumerate(jobs, start=1):
        mode = "alert-only" if job.alert_only else (
            job.ats_provider.value if job.ats_provider else "ats"
        )
        score = job.llm_score or "-"
        missing = ", ".join(job.missing_skills[:4]) if job.missing_skills else "none"
        lines.append(
            f"{idx}. [{score}/5] {job.company_name} — {job.title} ({mode})\n"
            f"   ID: #{job.id} | Location: {job.location_raw or 'Unspecified'}\n"
            f"   Why: {job.llm_reasoning or 'N/A'}\n"
            f"   Gaps: {missing}\n"
            f"   Apply: {job.apply_url}"
        )
    return "\n".join(lines)


def render_digest_table(jobs: list[JobRecord], console: Console) -> None:
    """Render ranked digest jobs as a Rich terminal table."""
    if not jobs:
        console.print("[yellow]No scored jobs match the digest threshold.[/yellow]")
        return

    table = Table(title=f"Doot Ranked Digest ({len(jobs)} jobs)")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Company", style="bold")
    table.add_column("Role")
    table.add_column("Location")
    table.add_column("Mode")
    table.add_column("Status")

    for job in jobs:
        mode = "alert-only" if job.alert_only else (
            job.ats_provider.value if job.ats_provider else "ats"
        )
        score_str = f"{job.llm_score}/5" if job.llm_score is not None else "-"
        table.add_row(
            f"#{job.id}",
            score_str,
            job.company_name,
            job.title,
            job.location_raw or "-",
            mode,
            job.status.value,
        )
    console.print(table)


def render_job_panel(job: JobRecord, console: Console) -> None:
    """Render a detailed Rich panel for interactive CLI review."""
    mode = "alert-only" if job.alert_only else (
        job.ats_provider.value if job.ats_provider else "ats"
    )
    missing = ", ".join(job.missing_skills) if job.missing_skills else "None identified"
    body = (
        f"[bold]{job.title}[/bold] at [cyan]{job.company_name}[/cyan]\n"
        f"Score: [green]{job.llm_score or '-'}/5[/green] | "
        f"Mode: {mode} | Status: {job.status.value}\n"
        f"Location: {job.location_raw or 'Unspecified'}\n"
        f"URL: {job.apply_url}\n\n"
        f"[bold]Reasoning:[/bold] {job.llm_reasoning or 'N/A'}\n"
        f"[bold]Missing Skills:[/bold] {missing}"
    )
    console.print(Panel(body, title=f"Job #{job.id}", border_style="cyan"))
