"""Anti-bot, CAPTCHA, and email-verification challenge detector."""

import re

_CHALLENGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "reCAPTCHA",
        re.compile(r"(?:g-recaptcha|google\.com/recaptcha|recaptcha/api)", re.IGNORECASE),
    ),
    ("hCaptcha", re.compile(r"(?:h-captcha|hcaptcha\.com)", re.IGNORECASE)),
    (
        "Cloudflare Turnstile / Challenge",
        re.compile(
            r"(?:cf-turnstile|challenges\.cloudflare\.com|"
            r"verify you are human|just a moment\.\.\.)",
            re.IGNORECASE,
        ),
    ),
    (
        "Email Verification / Security Code",
        re.compile(
            r"(?:enter (?:the )?(?:verification|security) code|code sent to your email|"
            r"greenhouse security code|confirm your email address with a code)",
            re.IGNORECASE,
        ),
    ),
)


def detect_anti_bot_challenge(
    html_or_text: str,
    *,
    frame_urls: list[str] | None = None,
) -> str | None:
    """Detect CAPTCHAs or email security gates; returns challenge name or None.

    Automation must stop immediately whenever a challenge is detected.
    """
    haystack = html_or_text
    if frame_urls:
        haystack = f"{haystack}\n" + "\n".join(frame_urls)

    for label, pattern in _CHALLENGE_PATTERNS:
        if pattern.search(haystack):
            return label
    return None
