"""Offline fixture tests for ATS detectors, rate limiter, and JSON collectors."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from doot.collectors.ats import (
    AshbyCollector,
    ATSNotFoundError,
    GreenhouseCollector,
    LeverCollector,
    detect_ats_from_html,
    detect_ats_from_url,
)
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models import ATSProvider, CompanyRecord

FIXTURES = Path(__file__).parent / "fixtures"


def test_detect_ats_from_various_urls_and_html() -> None:
    gh = detect_ats_from_url("https://job-boards.greenhouse.io/postman/jobs/554433")
    assert gh is not None
    assert gh.ats_provider == ATSProvider.GREENHOUSE
    assert gh.board_token == "postman"
    assert gh.external_job_id == "554433"

    gh_embed = detect_ats_from_url("https://boards.greenhouse.io/embed/job_app?for=acme&token=77")
    assert gh_embed is not None
    assert gh_embed.board_token == "acme"
    assert gh_embed.external_job_id == "77"

    lever = detect_ats_from_url("https://jobs.lever.co/cred/1122-3344/apply")
    assert lever is not None
    assert lever.ats_provider == ATSProvider.LEVER
    assert lever.board_token == "cred"
    assert lever.external_job_id == "1122-3344"

    ashby = detect_ats_from_url("https://jobs.ashbyhq.com/gamma-ai/abc-def/application")
    assert ashby is not None
    assert ashby.ats_provider == ATSProvider.ASHBY
    assert ashby.board_token == "gamma-ai"
    assert ashby.external_job_id == "abc-def"

    html_text = (FIXTURES / "careers_embed.html").read_text(encoding="utf-8")
    embedded = detect_ats_from_html(html_text)
    assert embedded is not None
    assert embedded.ats_provider == ATSProvider.GREENHOUSE
    assert embedded.board_token == "deltaembedded"


async def test_domain_rate_limiter_enforces_interval() -> None:
    now = 100.0
    slept: list[float] = []

    def fake_time() -> float:
        return now

    async def fake_sleep(dt: float) -> None:
        nonlocal now
        slept.append(dt)
        now += dt

    limiter = DomainRateLimiter(
        min_interval_seconds=2.0,
        jitter_range=(0.0, 0.0),
        time_fn=fake_time,
        sleep_fn=fake_sleep,
    )
    await limiter.acquire("https://boards-api.greenhouse.io/v1/boards/a/jobs")
    await limiter.acquire("https://boards-api.greenhouse.io/v1/boards/b/jobs")
    assert slept == [2.0]


@respx.mock
async def test_greenhouse_lever_ashby_collectors_from_fixtures() -> None:
    gh_payload = json.loads((FIXTURES / "greenhouse_jobs.json").read_text(encoding="utf-8"))
    lever_payload = json.loads((FIXTURES / "lever_jobs.json").read_text(encoding="utf-8"))
    ashby_payload = json.loads((FIXTURES / "ashby_jobs.json").read_text(encoding="utf-8"))

    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true").mock(
        return_value=httpx.Response(200, json=gh_payload)
    )
    respx.get("https://api.lever.co/v0/postings/beta?mode=json").mock(
        return_value=httpx.Response(200, json=lever_payload)
    )
    respx.get("https://api.ashbyhq.com/posting-api/job-board/gamma-ai").mock(
        return_value=httpx.Response(200, json=ashby_payload)
    )

    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    async with httpx.AsyncClient() as client:
        gh_jobs = await GreenhouseCollector(
            CompanyRecord(
                id=1,
                name="Acme",
                normalized_name="acme",
                ats_provider=ATSProvider.GREENHOUSE,
                board_token="acme",
            ),
            client,
            limiter,
        ).fetch()
        assert len(gh_jobs) == 2
        assert "event-driven AWS Lambda systems." in gh_jobs[0].jd_text
        assert gh_jobs[0].alert_only is False

        lever_jobs = await LeverCollector(
            CompanyRecord(
                id=2,
                name="Beta",
                normalized_name="beta",
                ats_provider=ATSProvider.LEVER,
                board_token="beta",
            ),
            client,
            limiter,
        ).fetch()
        assert len(lever_jobs) == 1
        assert "Requirements" in lever_jobs[0].jd_text
        assert "AWS SQS" in lever_jobs[0].jd_text

        ashby_jobs = await AshbyCollector(
            CompanyRecord(
                id=3,
                name="Gamma AI",
                normalized_name="gamma ai",
                ats_provider=ATSProvider.ASHBY,
                board_token="gamma-ai",
            ),
            client,
            limiter,
        ).fetch()
        assert len(ashby_jobs) == 1
        assert ashby_jobs[0].salary_raw == "25 - 35 LPA"
        assert "Remote" in ashby_jobs[0].location_raw


@respx.mock
async def test_ats_404_raises_not_found() -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/missing/jobs?content=true").mock(
        return_value=httpx.Response(404)
    )
    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    async with httpx.AsyncClient() as client:
        collector = GreenhouseCollector(
            CompanyRecord(
                id=99,
                name="Missing",
                normalized_name="missing",
                ats_provider=ATSProvider.GREENHOUSE,
                board_token="missing",
            ),
            client,
            limiter,
        )
        with pytest.raises(ATSNotFoundError):
            await collector.fetch()
