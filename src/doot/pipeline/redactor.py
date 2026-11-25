"""Deterministic PII redaction before sending candidate data to external LLMs."""

import re

from doot.models.profile import UserProfile

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")


def redact_pii(text: str, profile: UserProfile) -> str:
    """Scrub name, email, phone, and personal URLs from text before LLM prompts."""
    if not text:
        return ""

    redacted = text

    # Scrub exact links first so domain/username substrings aren't left intact
    for url in (profile.links.linkedin, profile.links.github, profile.links.portfolio):
        if url and url.strip():
            redacted = redacted.replace(url.strip(), "[REDACTED_URL]")

    if profile.email.strip():
        redacted = redacted.replace(profile.email.strip(), "[REDACTED_EMAIL]")

    if profile.phone.strip():
        phone_raw = profile.phone.strip()
        redacted = redacted.replace(phone_raw, "[REDACTED_PHONE]")
        digits_only = re.sub(r"\D", "", phone_raw)
        if len(digits_only) >= 8:
            redacted = redacted.replace(digits_only, "[REDACTED_PHONE]")
            redacted = redacted.replace(digits_only[-10:], "[REDACTED_PHONE]")

    if profile.full_name.strip():
        redacted = re.sub(
            re.escape(profile.full_name.strip()),
            "[REDACTED_NAME]",
            redacted,
            flags=re.IGNORECASE,
        )
        for token in profile.full_name.split():
            if len(token) >= 3:
                redacted = re.sub(
                    rf"\b{re.escape(token)}\b",
                    "[REDACTED_NAME]",
                    redacted,
                    flags=re.IGNORECASE,
                )

    # Generic fallback patterns
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    return redacted


def build_redacted_profile_summary(profile: UserProfile, resume_markdown: str = "") -> str:
    """Create an anonymized candidate context string safe to send to Gemini."""
    lines = [
        f"Years of experience: {profile.years_of_experience}",
        f"Preferred work modes: {', '.join(profile.preferred_work_modes) or 'remote, hybrid'}",
        f"Preferred locations: {', '.join(profile.preferred_locations) or 'India, Remote'}",
        f"Target roles: {', '.join(profile.target_roles) or 'Backend, Full Stack, AI'}",
        f"Work authorization (India): {profile.work_authorization_india}",
        f"Requires US visa sponsorship: {profile.requires_us_visa}",
    ]
    if resume_markdown.strip():
        lines.append("")
        lines.append("Candidate Resume (PII Redacted):")
        lines.append(redact_pii(resume_markdown, profile))
    return "\n".join(lines)
