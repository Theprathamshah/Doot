"""Global kill-switch and application volume/safety guardrails."""

from dataclasses import dataclass

from doot.config import DootSettings
from doot.db.repository import DootRepository


class SafetyGuardError(RuntimeError):
    """Raised when a safety guardrail blocks execution."""


@dataclass(frozen=True, slots=True)
class SafetyStatus:
    """Snapshot of current kill-switch and volume cap checks."""

    stop_file_active: bool
    applies_today: int
    daily_cap: int
    daily_cap_reached: bool


def assert_not_stopped(settings: DootSettings) -> None:
    """Abort immediately if the global `data/STOP` sentinel file exists."""
    if settings.stop_file_path.exists():
        raise SafetyGuardError(
            f"Global kill switch active: {settings.stop_file_path} exists. "
            "Remove it (or run `doot resume`) to continue."
        )


async def check_apply_guardrails(
    settings: DootSettings,
    repo: DootRepository,
    *,
    company_id: int | None = None,
    apply_flag: bool = False,
) -> None:
    """Enforce all non-negotiable safety invariants prior to any submission."""
    assert_not_stopped(settings)

    if settings.dry_run or not apply_flag:
        raise SafetyGuardError(
            "Submission blocked: dry-run mode is active. Pass --apply and set DOOT_DRY_RUN=false."
        )

    applies_today = await repo.count_automated_applies_today()
    if applies_today >= settings.daily_apply_cap:
        raise SafetyGuardError(
            f"Daily application cap reached ({applies_today}/{settings.daily_apply_cap})."
        )

    if company_id is not None:
        company_applies = await repo.count_automated_applies_for_company(company_id)
        if company_applies >= settings.per_company_cap:
            raise SafetyGuardError(
                "Per-company application cap reached "
                f"({company_applies}/{settings.per_company_cap})."
            )


async def get_safety_status(settings: DootSettings, repo: DootRepository) -> SafetyStatus:
    """Return non-throwing safety status for diagnostics."""
    applies_today = await repo.count_automated_applies_today()
    return SafetyStatus(
        stop_file_active=settings.stop_file_path.exists(),
        applies_today=applies_today,
        daily_cap=settings.daily_apply_cap,
        daily_cap_reached=applies_today >= settings.daily_apply_cap,
    )
