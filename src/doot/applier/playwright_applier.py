"""Headed Playwright assisted-apply runner for Greenhouse, Lever, and Ashby."""

import re
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import yaml
from bs4 import BeautifulSoup, Tag
from loguru import logger
from playwright.async_api import async_playwright

from doot.applier.base import FilledFormResult, FormField, SubmitResult
from doot.applier.challenge import detect_anti_bot_challenge
from doot.applier.field_resolver import FieldResolver
from doot.collectors.ats.detector import detect_ats_from_html, detect_ats_from_url
from doot.config import DootSettings
from doot.db.repository import DootRepository
from doot.kill_switch import SafetyGuardError, assert_not_stopped, check_apply_guardrails
from doot.models.enums import ATSProvider, JobStatus
from doot.models.job import JobPosting, JobRecord, TailoredAssets
from doot.models.profile import UserProfile

_SUPPORTED_ATS = frozenset({ATSProvider.GREENHOUSE, ATSProvider.LEVER, ATSProvider.ASHBY})
ConfirmFn = Callable[[FilledFormResult], Awaitable[bool]]


class PageLike(Protocol):
    """Minimal async Playwright Page interface for testability."""

    @property
    def frames(self) -> list[Any]: ...

    async def goto(self, url: str, **kwargs: Any) -> Any: ...

    async def content(self) -> str: ...

    async def screenshot(self, **kwargs: Any) -> Any: ...

    def get_by_label(self, text: Any, **kwargs: Any) -> Any: ...

    def get_by_placeholder(self, text: Any, **kwargs: Any) -> Any: ...

    def get_by_role(self, role: Any, **kwargs: Any) -> Any: ...

    def locator(self, selector: str) -> Any: ...


PageSessionFactory = Callable[[], AbstractAsyncContextManager[PageLike]]


def load_selectors_config(path: Path) -> dict[str, dict[str, str]]:
    """Load per-ATS selector overrides from selectors.yaml."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_ats = data.get("ats") or {}
    return {
        str(k): {str(sk): str(sv) for sk, sv in v.items()}
        for k, v in raw_ats.items()
        if isinstance(v, dict)
    }


def resolve_canonical_apply_url(apply_url: str, page_html: str = "") -> str:
    """Resolve embedded ATS iframes/urls to their canonical direct apply URL."""
    detected = detect_ats_from_url(apply_url)
    if detected is None and page_html:
        detected = detect_ats_from_html(page_html)
    if detected is not None and detected.canonical_url:
        return detected.canonical_url
    return apply_url


def extract_form_fields_from_html(html_text: str) -> list[FormField]:
    """Extract visible form inputs and their accessible labels from ATS HTML."""
    soup = BeautifulSoup(html_text, "html.parser")
    labels_by_for: dict[str, str] = {}
    for lbl in soup.find_all("label"):
        for_id = lbl.get("for")
        text = " ".join(lbl.get_text(" ", strip=True).split())
        if isinstance(for_id, str) and for_id and text:
            labels_by_for[for_id] = text

    fields: list[FormField] = []
    seen_labels: set[str] = set()

    for tag in soup.find_all(["input", "textarea", "select"]):
        if not isinstance(tag, Tag):
            continue
        tag_name = tag.name.lower()
        input_type = str(tag.get("type") or "text").lower() if tag_name == "input" else tag_name
        if input_type in {"hidden", "submit", "button", "reset", "image"}:
            continue

        el_id = str(tag.get("id") or "").strip()
        el_name = str(tag.get("name") or "").strip()
        aria_label = str(tag.get("aria-label") or "").strip()
        placeholder = str(tag.get("placeholder") or "").strip()

        label = labels_by_for.get(el_id, "") or aria_label
        if not label:
            parent_label = tag.find_parent("label")
            if isinstance(parent_label, Tag):
                label = " ".join(parent_label.get_text(" ", strip=True).split())
        if not label:
            label = placeholder or el_name or el_id

        label = label.rstrip("*").strip()
        if not label or label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())

        selector_hint = None
        if el_id:
            selector_hint = f"#{el_id}"
        elif el_name:
            selector_hint = f'{tag_name}[name="{el_name}"]'

        required = tag.has_attr("required") or tag.get("aria-required") == "true"
        fields.append(
            FormField(
                label=label,
                field_type=input_type,
                required=required,
                selector_hint=selector_hint,
            )
        )
    return fields


@asynccontextmanager
async def _default_headed_page_session(headless: bool = False):  # type: ignore[no-untyped-def]
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            yield page
        finally:
            await context.close()
            await browser.close()


async def _transition_to_prepared(repo: DootRepository, job: JobRecord) -> None:
    if job.status == JobStatus.SCORED:
        await repo.transition_job(
            job.id,
            JobStatus.NOTIFIED,
            actor="applier",
            reason="surfaced for assisted apply",
        )
        await repo.transition_job(
            job.id,
            JobStatus.PREPARED,
            actor="applier",
            reason="started form preparation",
        )
    elif job.status in (
        JobStatus.NOTIFIED,
        JobStatus.AWAITING_APPROVAL,
        JobStatus.FAILED,
        JobStatus.NEEDS_MANUAL_ACTION,
    ):
        await repo.transition_job(
            job.id,
            JobStatus.PREPARED,
            actor="applier",
            reason="started form preparation",
        )


async def _fill_locator(page: PageLike, field: FormField, value: str) -> bool:
    for candidate in (
        page.get_by_label(field.label, exact=False),
        page.get_by_placeholder(field.label, exact=False),
        page.locator(field.selector_hint) if field.selector_hint else None,
    ):
        if candidate is None:
            continue
        try:
            loc = candidate.first
            if await loc.count() == 0:
                continue
            if field.field_type == "select":
                await loc.select_option(label=value)
            else:
                await loc.fill(value)
            return True
        except Exception:
            continue
    return False


async def run_assisted_apply(
    settings: DootSettings,
    repo: DootRepository,
    *,
    job_id: int,
    profile: UserProfile,
    field_resolver: FieldResolver,
    tailored_assets: TailoredAssets | None = None,
    apply_flag: bool = False,
    headless: bool = False,
    confirm_submit_fn: ConfirmFn | None = None,
    page_factory: PageSessionFactory | None = None,
) -> SubmitResult:
    """Open headed browser, pre-fill Greenhouse/Lever/Ashby form, and submit only on approval."""
    assert_not_stopped(settings)

    job = await repo.get_job(job_id)
    if job is None:
        raise LookupError(f"Job #{job_id} not found")
    if job.alert_only:
        raise SafetyGuardError(
            f"Job #{job_id} ({job.source}) is alert-only; automated apply is forbidden."
        )
    if job.ats_provider not in _SUPPORTED_ATS:
        raise SafetyGuardError(
            f"Unsupported ATS '{job.ats_provider}' for automated fill "
            "(Greenhouse/Lever/Ashby only)."
        )

    await _transition_to_prepared(repo, job)

    selectors = load_selectors_config(settings.data_dir / "selectors.yaml")
    ats_selectors = selectors.get(job.ats_provider.value, {})
    posting = JobPosting.model_validate(job.model_dump())

    session_cm = (
        page_factory()
        if page_factory is not None
        else _default_headed_page_session(headless=headless)
    )

    async with session_cm as page:
        target_url = resolve_canonical_apply_url(job.apply_url)
        await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
        html_text = await page.content()
        frame_urls = [getattr(f, "url", "") for f in getattr(page, "frames", [])]

        challenge = detect_anti_bot_challenge(html_text, frame_urls=frame_urls)
        if challenge is not None:
            msg = f"Anti-bot challenge detected ({challenge}); finish manually in browser."
            await repo.mark_application_failed(job_id, msg, status="NEEDS_MANUAL_ACTION")
            await repo.transition_job(
                job_id,
                JobStatus.NEEDS_MANUAL_ACTION,
                actor="applier",
                reason=msg,
            )
            return SubmitResult(submitted=False, status=JobStatus.NEEDS_MANUAL_ACTION, message=msg)

        fields = extract_form_fields_from_html(html_text)
        resolved: dict[str, str] = {}
        unknown: list[str] = []

        for field in fields:
            if field.field_type == "file":
                if profile.resume_pdf_path.exists():
                    selector = (
                        ats_selectors.get("resume_input")
                        or field.selector_hint
                        or 'input[type="file"]'
                    )
                    await page.locator(selector).first.set_input_files(
                        str(profile.resume_pdf_path)
                    )
                    resolved[field.label] = str(profile.resume_pdf_path)
                else:
                    unknown.append(
                        f"{field.label} (missing resume PDF at {profile.resume_pdf_path})"
                    )
                continue

            answer = await field_resolver.resolve_field(
                field.label,
                job=posting,
                tailored_assets=tailored_assets,
            )
            if answer is None:
                if field.required:
                    unknown.append(field.label)
                continue

            filled = await _fill_locator(page, field, answer)
            if filled:
                resolved[field.label] = answer

        screenshots_dir = settings.data_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshots_dir / f"job_{job_id}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)

        prepared = FilledFormResult(
            resolved_fields=resolved,
            unknown_questions=unknown,
            screenshot_path=str(screenshot_path),
            challenge_detected=None,
        )

        await repo.save_application_prepared(
            job_id,
            filled_fields=resolved,
            screenshot_path=str(screenshot_path),
            resume_path=str(profile.resume_pdf_path),
            status="AWAITING_APPROVAL",
        )
        await repo.transition_job(
            job_id,
            JobStatus.AWAITING_APPROVAL,
            actor="applier",
            reason=f"filled {len(resolved)} field(s); unknown={len(unknown)}",
        )

        try:
            await check_apply_guardrails(
                settings,
                repo,
                company_id=job.company_id,
                apply_flag=apply_flag,
            )
        except SafetyGuardError as exc:
            return SubmitResult(
                submitted=False,
                status=JobStatus.AWAITING_APPROVAL,
                message=f"Prepared form (not submitted): {exc}",
            )

        approved = await confirm_submit_fn(prepared) if confirm_submit_fn is not None else False
        if not approved:
            return SubmitResult(
                submitted=False,
                status=JobStatus.AWAITING_APPROVAL,
                message="Prepared form; submission cancelled by user.",
            )

        submit_selector = ats_selectors.get("submit_button") or 'button[type="submit"]'
        submit_loc = page.locator(submit_selector).first
        if await submit_loc.count() > 0:
            await submit_loc.click()
        else:
            await page.get_by_role("button", name=re.compile("submit", re.IGNORECASE)).first.click()

        await repo.mark_application_submitted(job_id)
        await repo.transition_job(
            job_id,
            JobStatus.APPLIED,
            actor="applier",
            reason="submitted after human approval",
        )
        logger.info("Submitted application for job #{} ({})", job_id, job.company_name)
        return SubmitResult(
            submitted=True,
            status=JobStatus.APPLIED,
            message=f"Application submitted for job #{job_id}.",
        )
