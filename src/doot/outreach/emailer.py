"""Grounded cold-email drafting and safe rate-limited SMTP delivery."""

import asyncio
import smtplib
from collections.abc import Awaitable, Callable
from email.message import EmailMessage
from pathlib import Path

from pydantic import BaseModel

from doot.config import DootSettings
from doot.db.repository import DootRepository
from doot.kill_switch import SafetyGuardError, assert_not_stopped
from doot.llm.base import LLMProvider
from doot.llm.gemini import DailyQuotaExceededError
from doot.models.job import JobPosting, JobRecord
from doot.models.profile import UserProfile
from doot.rag import BM25Retriever, collect_evidence_chunks

SendMailFn = Callable[[DootSettings, str, str, str, Path | None], None]


class ColdEmailDraft(BaseModel):
    """Personalized cold outreach email for HR, Engineering Manager, or Founder."""

    subject: str
    body: str


def _template_cold_email(
    *,
    profile: UserProfile,
    company_name: str,
    role_title: str,
    person_name: str,
    evidence_snippets: list[str],
) -> ColdEmailDraft:
    greeting = f"Hi {person_name.split()[0]}," if person_name.strip() else "Hi there,"
    highlight = (
        evidence_snippets[0].splitlines()[-1]
        if evidence_snippets
        else "built event-driven AWS Lambda & Java/Node microservices handling 100K+ daily orders"
    )
    yoe = f"{profile.years_of_experience:g}"
    subject = f"{yoe}+ YOE Backend/Fullstack Engineer — {role_title} at {company_name}"
    lead = highlight[0].lower() + highlight[1:] if len(highlight) > 1 else highlight
    body = (
        f"{greeting}\n\n"
        f"I came across the {role_title} opening at {company_name} and wanted to reach out "
        f"directly. I'm a Full Stack / Backend Engineer ({yoe} YOE) working with "
        f"Java (Spring Boot), Node.js/TypeScript, React, and AWS.\n\n"
        f"Most recently, I {lead}\n\n"
        f"I'd love to share how I can contribute to {company_name}. "
        f"Happy to send over any extra details—my resume is attached.\n\n"
        f"Best,\n{profile.full_name}\n"
        f"{profile.links.linkedin or profile.links.github}"
    )
    return ColdEmailDraft(subject=subject[:140], body=body)


async def draft_cold_email(
    *,
    profile: UserProfile,
    company_name: str,
    role_title: str,
    person_name: str = "",
    recipient_role: str = "",
    job: JobRecord | None = None,
    resume_markdown: str = "",
    stories_markdown: str = "",
    llm_provider: LLMProvider | None = None,
) -> ColdEmailDraft:
    """Draft a grounded cold email citing real evidence (Gemini with template fallback)."""
    chunks = collect_evidence_chunks(
        profile, resume_markdown=resume_markdown, stories_markdown=stories_markdown
    )
    query = f"{role_title} {job.jd_text[:1500] if job else company_name}"
    selected = BM25Retriever(chunks).retrieve(query, top_k=3) if chunks else []
    snippets = [c.render_for_prompt() for c in selected]

    if llm_provider is not None and snippets and job is not None:
        try:
            posting = JobPosting.model_validate(job.model_dump())
            prompt = (
                f"Write a short, polite, high-signal cold email (<110 words) to "
                f"{person_name or 'the hiring team'} ({recipient_role or 'Hiring/Engineering'}) "
                f"at {company_name} for '{role_title}'. Cite ONLY facts from the evidence. "
                f"Sign off as {profile.full_name}."
            )
            body = await llm_provider.draft_free_text_answer(prompt, snippets, posting)
            if body.strip():
                yoe = f"{profile.years_of_experience:g}"
                subject = f"{role_title} at {company_name} — {profile.full_name} ({yoe} YOE)"
                return ColdEmailDraft(subject=subject[:140], body=body.strip())
        except (DailyQuotaExceededError, NotImplementedError, Exception):
            pass

    return _template_cold_email(
        profile=profile,
        company_name=company_name,
        role_title=role_title,
        person_name=person_name,
        evidence_snippets=[c.content for c in selected],
    )


def _default_send_smtp(
    settings: DootSettings,
    to_email: str,
    subject: str,
    body: str,
    attachment_path: Path | None,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.notify_email_from
    msg["To"] = to_email
    msg.set_content(body)

    if attachment_path is not None and attachment_path.exists():
        msg.add_attachment(
            attachment_path.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=attachment_path.name,
        )

    with smtplib.SMTP(settings.notify_smtp_host, settings.notify_smtp_port, timeout=20) as smtp:
        smtp.starttls()
        password = settings.notify_email_password.get_secret_value()
        if password:
            smtp.login(settings.notify_email_from, password)
        smtp.send_message(msg)


async def send_outreach_email(
    settings: DootSettings,
    repo: DootRepository,
    *,
    contact_id: int,
    draft: ColdEmailDraft,
    job_id: int | None = None,
    send_flag: bool = False,
    attachment_path: Path | None = None,
    confirm_fn: Callable[[ColdEmailDraft, str], Awaitable[bool]] | None = None,
    send_fn: SendMailFn | None = None,
) -> bool:
    """Draft or send a cold email with kill-switch, daily cap, and dry-run guards."""
    assert_not_stopped(settings)
    contact = await repo.get_contact(contact_id)
    if contact is None:
        raise LookupError(f"Contact #{contact_id} not found")
    if contact.get("emailed_at"):
        raise SafetyGuardError(
            f"Contact #{contact_id} ({contact['email']}) was already emailed "
            f"at {contact['emailed_at']}."
        )

    if not send_flag or settings.dry_run:
        await repo.record_outreach_email(
            contact_id=contact_id,
            job_id=job_id,
            subject=draft.subject,
            body=draft.body,
            sent=False,
        )
        return False

    sent_today = await repo.count_outreach_sent_today()
    if sent_today >= settings.daily_outreach_cap:
        raise SafetyGuardError(
            f"Daily outreach cap reached ({sent_today}/{settings.daily_outreach_cap})."
        )

    if not (settings.notify_smtp_host and settings.notify_email_from):
        raise SafetyGuardError(
            "SMTP settings missing. Set DOOT_NOTIFY_SMTP_HOST, DOOT_NOTIFY_EMAIL_FROM, "
            "and DOOT_NOTIFY_EMAIL_PASSWORD in .env."
        )

    approved = await confirm_fn(draft, str(contact["email"])) if confirm_fn else False
    if not approved:
        await repo.record_outreach_email(
            contact_id=contact_id,
            job_id=job_id,
            subject=draft.subject,
            body=draft.body,
            sent=False,
        )
        return False

    mailer = send_fn or _default_send_smtp
    try:
        await asyncio.to_thread(
            mailer,
            settings,
            str(contact["email"]),
            draft.subject,
            draft.body,
            attachment_path,
        )
    except Exception as exc:
        await repo.record_outreach_email(
            contact_id=contact_id,
            job_id=job_id,
            subject=draft.subject,
            body=draft.body,
            sent=False,
            error_message=str(exc)[:400],
        )
        raise

    await repo.record_outreach_email(
        contact_id=contact_id,
        job_id=job_id,
        subject=draft.subject,
        body=draft.body,
        sent=True,
    )
    return True
