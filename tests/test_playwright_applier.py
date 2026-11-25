"""Offline HTML fixture tests for Playwright form extraction, dry-run, CAPTCHA, and submit."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from doot.applier import (
    FieldResolver,
    extract_form_fields_from_html,
    load_aliases_config,
    resolve_canonical_apply_url,
    run_assisted_apply,
)
from doot.config import DootSettings
from doot.db import DootRepository
from doot.kill_switch import SafetyGuardError
from doot.models import ATSProvider, JobStatus, ProfileLinks, UserProfile
from tests.conftest import make_sample_job

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeLocator:
    def __init__(self, key: str, state: dict[str, Any]) -> None:
        self.key = key
        self.state = state

    @property
    def first(self) -> "_FakeLocator":
        return self

    async def count(self) -> int:
        return 1

    async def fill(self, value: str) -> None:
        self.state["filled"][self.key] = value

    async def select_option(self, **kwargs: Any) -> None:
        self.state["filled"][self.key] = str(kwargs)

    async def set_input_files(self, path: str) -> None:
        self.state["uploaded"].append(path)

    async def click(self) -> None:
        self.state["clicked"].append(self.key)


class _FakePage:
    def __init__(self, html_text: str) -> None:
        self.html_text = html_text
        self.state: dict[str, Any] = {"filled": {}, "uploaded": [], "clicked": [], "url": None}

    @property
    def frames(self) -> list[Any]:
        return []

    async def goto(self, url: str, **kwargs: Any) -> None:
        self.state["url"] = url

    async def content(self) -> str:
        return self.html_text

    async def screenshot(self, **kwargs: Any) -> None:
        path = Path(str(kwargs["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-png")

    def get_by_label(self, text: Any, **kwargs: Any) -> _FakeLocator:
        return _FakeLocator(f"label:{text}", self.state)

    def get_by_placeholder(self, text: Any, **kwargs: Any) -> _FakeLocator:
        return _FakeLocator(f"placeholder:{text}", self.state)

    def get_by_role(self, role: Any, **kwargs: Any) -> _FakeLocator:
        return _FakeLocator(f"role:{role}", self.state)

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(f"selector:{selector}", self.state)


def _profile(tmp_settings: DootSettings) -> UserProfile:
    resume_pdf = tmp_settings.data_dir / "resume.pdf"
    resume_pdf.write_bytes(b"%PDF-1.4\n")
    return UserProfile(
        full_name="Pratham Shah",
        email="pratham@example.com",
        phone="+91-9876543210",
        location="Bengaluru, India",
        work_authorization_india="Authorized in India",
        years_of_experience=2.5,
        current_ctc_lpa=12.0,
        expected_ctc_lpa=22.0,
        notice_period_days=30,
        links=ProfileLinks(
            linkedin="https://linkedin.com/in/pratham",
            github="https://github.com/pratham",
        ),
        resume_pdf_path=resume_pdf,
    )


def test_extract_form_fields_from_ats_fixtures_and_iframe_url() -> None:
    gh_fields = extract_form_fields_from_html(
        (FIXTURES / "form_greenhouse.html").read_text(encoding="utf-8")
    )
    gh_labels = {f.label for f in gh_fields}
    assert {"First Name", "Last Name", "Email", "Resume/CV", "Expected CTC (LPA)"}.issubset(
        gh_labels
    )

    lever_fields = extract_form_fields_from_html(
        (FIXTURES / "form_lever.html").read_text(encoding="utf-8")
    )
    assert any(f.label == "LinkedIn URL" for f in lever_fields)

    ashby_fields = extract_form_fields_from_html(
        (FIXTURES / "form_ashby.html").read_text(encoding="utf-8")
    )
    assert any(f.label == "Years of experience" for f in ashby_fields)

    embed_html = (FIXTURES / "careers_embed.html").read_text(encoding="utf-8")
    assert (
        resolve_canonical_apply_url("https://delta.example.com/careers", embed_html)
        == "https://boards.greenhouse.io/deltaembedded"
    )


async def test_assisted_apply_blocks_alert_only_and_halts_on_captcha(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    profile = _profile(tmp_settings)
    aliases = load_aliases_config(Path(__file__).resolve().parent.parent / "data" / "aliases.yaml")
    resolver = FieldResolver(profile=profile, aliases=aliases, repo=repo)

    # 1) Alert-only LinkedIn job must be blocked
    li_job = make_sample_job(external_job_id="li-1", source="jobspy_linkedin", alert_only=True)
    li_id, _ = await repo.upsert_job(li_job)
    with pytest.raises(SafetyGuardError, match="alert-only"):
        await run_assisted_apply(
            tmp_settings,
            repo,
            job_id=li_id,
            profile=profile,
            field_resolver=resolver,
        )

    # 2) CAPTCHA page -> stop immediately and mark NEEDS_MANUAL_ACTION
    gh_job = make_sample_job(external_job_id="gh-cap")
    gh_id, _ = await repo.upsert_job(gh_job)
    await repo.transition_job(gh_id, JobStatus.SCORED, actor="test")
    captcha_html = (FIXTURES / "form_captcha.html").read_text(encoding="utf-8")
    fake_page = _FakePage(captcha_html)

    @asynccontextmanager
    async def _page_factory():  # type: ignore[no-untyped-def]
        yield fake_page

    res = await run_assisted_apply(
        tmp_settings,
        repo,
        job_id=gh_id,
        profile=profile,
        field_resolver=resolver,
        page_factory=_page_factory,
    )
    assert res.submitted is False
    assert res.status == JobStatus.NEEDS_MANUAL_ACTION
    assert fake_page.state["clicked"] == []


async def test_assisted_apply_dry_run_then_live_submit_with_cap(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    profile = _profile(tmp_settings)
    aliases = load_aliases_config(Path(__file__).resolve().parent.parent / "data" / "aliases.yaml")
    resolver = FieldResolver(profile=profile, aliases=aliases, repo=repo)

    cid = await repo.upsert_company(
        name="Acme Corp",
        normalized_name="acme corp",
        ats_provider=ATSProvider.GREENHOUSE,
        board_token="acme",
    )
    j1_id, _ = await repo.upsert_job(make_sample_job(company_id=cid, external_job_id="gh-1"))
    j2_id, _ = await repo.upsert_job(make_sample_job(company_id=cid, external_job_id="gh-2"))
    await repo.transition_job(j1_id, JobStatus.SCORED, actor="test")
    await repo.transition_job(j2_id, JobStatus.SCORED, actor="test")

    gh_html = (FIXTURES / "form_greenhouse.html").read_text(encoding="utf-8")
    page1 = _FakePage(gh_html)

    @asynccontextmanager
    async def _factory1():  # type: ignore[no-untyped-def]
        yield page1

    # Dry-run fills fields, uploads resume, saves screenshot, does NOT click submit
    dry_res = await run_assisted_apply(
        tmp_settings,
        repo,
        job_id=j1_id,
        profile=profile,
        field_resolver=resolver,
        apply_flag=False,
        page_factory=_factory1,
    )
    assert dry_res.submitted is False
    assert dry_res.status == JobStatus.AWAITING_APPROVAL
    assert page1.state["filled"]["label:Email"] == "pratham@example.com"
    assert page1.state["filled"]["label:Expected CTC (LPA)"] == "22 LPA"
    assert len(page1.state["uploaded"]) == 1
    assert page1.state["clicked"] == []
    assert (tmp_settings.data_dir / "screenshots" / f"job_{j1_id}.png").exists()

    # Live submit with explicit confirmation -> submits and marks APPLIED
    live_settings = tmp_settings.model_copy(
        update={"dry_run": False, "per_company_cap": 1, "daily_apply_cap": 15}
    )

    async def _approve(_: object) -> bool:
        return True

    live_res = await run_assisted_apply(
        live_settings,
        repo,
        job_id=j1_id,
        profile=profile,
        field_resolver=resolver,
        apply_flag=True,
        confirm_submit_fn=_approve,
        page_factory=_factory1,
    )
    assert live_res.submitted is True
    assert live_res.status == JobStatus.APPLIED
    assert len(page1.state["clicked"]) == 1

    # Second application at same company must be blocked by per_company_cap=1
    page2 = _FakePage(gh_html)

    @asynccontextmanager
    async def _factory2():  # type: ignore[no-untyped-def]
        yield page2

    capped_res = await run_assisted_apply(
        live_settings,
        repo,
        job_id=j2_id,
        profile=profile,
        field_resolver=resolver,
        apply_flag=True,
        confirm_submit_fn=_approve,
        page_factory=_factory2,
    )
    assert capped_res.submitted is False
    assert "Per-company application cap reached" in capped_res.message
    assert page2.state["clicked"] == []
