"""Offline tests for Reddit [Hiring] collector, contact finder, and cold outreach."""

import httpx
import pytest
import respx

from doot.collectors.base import SearchHit
from doot.collectors.feeds import RedditHiringCollector
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.config import DootSettings
from doot.db import DootRepository
from doot.kill_switch import SafetyGuardError
from doot.models import ProfileLinks, UserProfile
from doot.outreach import (
    draft_cold_email,
    extract_emails_from_text,
    find_contacts_for_company,
    generate_email_candidates,
    parse_linkedin_people_hits,
    send_outreach_email,
)


def _profile(tmp_settings: DootSettings) -> UserProfile:
    pdf = tmp_settings.data_dir / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return UserProfile(
        full_name="Pratham Shah",
        email="pratham@example.com",
        phone="+91-9876543210",
        location="Ahmedabad, India",
        work_authorization_india="Authorized in India",
        years_of_experience=2.5,
        current_ctc_lpa=12.0,
        expected_ctc_lpa=22.0,
        notice_period_days=30,
        links=ProfileLinks(linkedin="https://linkedin.com/in/pratham"),
        resume_pdf_path=pdf,
    )


@respx.mock
async def test_reddit_collector_filters_for_hire_and_spam() -> None:
    respx.get("https://www.reddit.com/r/remotejs/new.json?limit=25").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "r1",
                                "title": "[Hiring] Senior Node + AWS Backend Engineer (Remote)",
                                "selftext": "Apply at https://jobs.ashbyhq.com/coolco/1234",
                                "permalink": "/r/remotejs/comments/r1",
                            }
                        },
                        {
                            "data": {
                                "id": "r2",
                                "title": "[For Hire] React developer looking for work",
                                "selftext": "Hire me please",
                                "permalink": "/r/remotejs/comments/r2",
                            }
                        },
                        {
                            "data": {
                                "id": "r3",
                                "title": "[Hiring] Quick task $5/hr",
                                "selftext": "Low pay gig",
                                "permalink": "/r/remotejs/comments/r3",
                            }
                        },
                    ]
                }
            },
        )
    )

    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    async with httpx.AsyncClient() as client:
        jobs = await RedditHiringCollector(
            client, limiter, subreddits=("remotejs",), limit_per_sub=25
        ).fetch()

    assert len(jobs) == 1
    assert jobs[0].external_job_id == "r1"
    assert jobs[0].board_token == "coolco"


async def test_contact_finder_and_cold_outreach_guards(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    assert extract_emails_from_text("Send CV to founders@coolco.io or noreply@coolco.io") == [
        "founders@coolco.io"
    ]
    assert generate_email_candidates("Priya Sharma", "coolco.io")[0] == "priya.sharma@coolco.io"

    hits = [
        SearchHit(
            url="https://www.linkedin.com/in/priya-sharma-123",
            title="Priya Sharma - Engineering Manager - CoolCo | LinkedIn",
        )
    ]
    assert parse_linkedin_people_hits(hits) == [
        ("Priya Sharma", "Engineering Manager", "https://www.linkedin.com/in/priya-sharma-123")
    ]

    class _FakeSearch:
        provider_name = "fake"

        async def search(self, query: str, *, days: int = 7, max_results: int = 20):
            if "linkedin.com/in" in query:
                return hits
            return []

    async def _mx_ok(_: str) -> bool:
        return True

    res = await find_contacts_for_company(
        tmp_settings,
        repo,
        company_name="CoolCo",
        domain="coolco.io",
        jd_text="Email founders@coolco.io to apply directly.",
        search_provider=_FakeSearch(),
        mx_check_fn=_mx_ok,
    )
    assert res.mx_valid is True
    assert res.contacts_added == 2

    saved = await repo.list_contacts()
    emails = {c["email"]: c for c in saved}
    assert emails["founders@coolco.io"]["confidence"] == "verified_jd"
    assert emails["priya.sharma@coolco.io"]["confidence"] == "pattern_mx"

    # Draft grounded cold email (local template fallback)
    profile = _profile(tmp_settings)
    draft = await draft_cold_email(
        profile=profile,
        company_name="CoolCo",
        role_title="Backend Engineer (AWS)",
        person_name="Priya Sharma",
        stories_markdown=(
            "### [story:lambda-30x] AWS Lambda\n"
            "Cut AWS Lambda latency 30x in production."
        ),
    )
    assert "Priya" in draft.body
    assert "30x" in draft.body

    priya_id = int(emails["priya.sharma@coolco.io"]["id"])

    # Dry-run records draft without sending
    sent_dry = await send_outreach_email(
        tmp_settings,
        repo,
        contact_id=priya_id,
        draft=draft,
        send_flag=False,
    )
    assert sent_dry is False

    # Live send with SMTP settings + confirm -> marks emailed_at and prevents duplicate send
    sent_calls: list[str] = []
    live_settings = tmp_settings.model_copy(
        update={
            "dry_run": False,
            "notify_smtp_host": "smtp.example.test",
            "notify_email_from": "pratham@example.com",
        }
    )

    async def _approve(_d: object, _r: str) -> bool:
        return True

    sent_live = await send_outreach_email(
        live_settings,
        repo,
        contact_id=priya_id,
        draft=draft,
        send_flag=True,
        attachment_path=profile.resume_pdf_path,
        confirm_fn=_approve,
        send_fn=lambda _s, to, _sub, _b, _att: sent_calls.append(to),
    )
    assert sent_live is True
    assert sent_calls == ["priya.sharma@coolco.io"]

    with pytest.raises(SafetyGuardError, match="already emailed"):
        await send_outreach_email(
            live_settings,
            repo,
            contact_id=priya_id,
            draft=draft,
            send_flag=True,
            confirm_fn=_approve,
        )
