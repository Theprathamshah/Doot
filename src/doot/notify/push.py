"""Optional one-way push notifications (macOS desktop, webhook, SMTP email)."""

import asyncio
import smtplib
from collections.abc import Awaitable, Callable
from email.message import EmailMessage

import httpx
from loguru import logger

from doot.config import DootSettings
from doot.models.job import JobRecord
from doot.notify.digest import build_digest_text

CommandRunner = Callable[[list[str]], Awaitable[int]]


async def _default_exec(cmd: list[str]) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await proc.wait()


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


async def send_macos_notification(
    title: str,
    message: str,
    *,
    runner: CommandRunner | None = None,
) -> bool:
    """Trigger a local macOS desktop notification via osascript."""
    exec_fn = runner or _default_exec
    script = (
        f'display notification "{_escape_applescript(message)}" '
        f'with title "{_escape_applescript(title)}"'
    )
    code = await exec_fn(["osascript", "-e", script])
    return code == 0


async def send_webhook_notification(
    webhook_url: str,
    summary_text: str,
    *,
    client: httpx.AsyncClient,
) -> bool:
    """Post a one-way digest summary to a generic/Discord/Slack-compatible webhook."""
    if not webhook_url.strip():
        return False
    resp = await client.post(
        webhook_url.strip(),
        json={"text": summary_text, "content": summary_text},
        timeout=15.0,
    )
    resp.raise_for_status()
    return True


def _send_smtp_sync(settings: DootSettings, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.notify_email_from
    msg["To"] = settings.notify_email_to
    msg.set_content(body)

    with smtplib.SMTP(settings.notify_smtp_host, settings.notify_smtp_port, timeout=20) as smtp:
        smtp.starttls()
        password = settings.notify_email_password.get_secret_value()
        if password:
            smtp.login(settings.notify_email_from, password)
        smtp.send_message(msg)


async def dispatch_one_way_alert(
    settings: DootSettings,
    jobs: list[JobRecord],
    *,
    client: httpx.AsyncClient | None = None,
    runner: CommandRunner | None = None,
) -> bool:
    """Send a one-way alert using settings.notify_method (`macos`, `webhook`, `email`, `none`)."""
    if not jobs:
        return False

    method = settings.notify_method.strip().lower()
    if method == "none":
        return False

    if method == "macos":
        top = jobs[0]
        body = (
            f"{len(jobs)} matching role(s). Top: {top.company_name} — {top.title} "
            f"[{top.llm_score or '-'}/5]. Run `doot review`."
        )
        return await send_macos_notification("Doot Job Digest", body, runner=runner)

    digest_text = build_digest_text(jobs)
    if method == "webhook":
        if not settings.notify_webhook_url.strip():
            logger.warning("DOOT_NOTIFY_METHOD=webhook but DOOT_NOTIFY_WEBHOOK_URL is empty.")
            return False
        if client is not None:
            return await send_webhook_notification(
                settings.notify_webhook_url, digest_text, client=client
            )
        async with httpx.AsyncClient() as owned_client:
            return await send_webhook_notification(
                settings.notify_webhook_url, digest_text, client=owned_client
            )

    if method == "email":
        if not (
            settings.notify_smtp_host
            and settings.notify_email_from
            and settings.notify_email_to
        ):
            logger.warning("Email notification skipped: SMTP settings incomplete.")
            return False
        subject = f"Doot Digest: {len(jobs)} matching role(s)"
        await asyncio.to_thread(_send_smtp_sync, settings, subject, digest_text)
        return True

    logger.warning("Unknown notify_method '{}'; skipping alert.", settings.notify_method)
    return False
