"""Detect ATS provider, board token, and job ID from URLs or HTML pages."""

import re
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel

from doot.models.enums import ATSProvider

_GREENHOUSE_HOSTS = frozenset(
    {
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "boards.eu.greenhouse.io",
        "job-boards.eu.greenhouse.io",
    }
)
_LEVER_HOSTS = frozenset({"jobs.lever.co", "jobs.eu.lever.co"})
_ASHBY_HOSTS = frozenset({"jobs.ashbyhq.com"})

_IGNORED_SEGMENTS = frozenset({"embed", "jobs", "apply", "application"})
_EMBED_JS_PATTERN = re.compile(
    r"https?://(?:boards|job-boards)(?:\.eu)?\.greenhouse\.io/embed/job_board(?:/js)?\?for=([a-zA-Z0-9_-]+)"
)


class DetectedATS(BaseModel):
    """Structured ATS identity extracted from a job or careers URL."""

    ats_provider: ATSProvider
    board_token: str
    external_job_id: str | None = None
    canonical_url: str | None = None


def _clean_path_parts(path: str) -> list[str]:
    return [part for part in path.strip("/").split("/") if part]


def detect_ats_from_url(url: str) -> DetectedATS | None:
    """Extract ATS provider, board token, and optional job ID from a URL."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if not host:
        return None

    parts = _clean_path_parts(parsed.path)
    qs = parse_qs(parsed.query)

    if host in _GREENHOUSE_HOSTS:
        # Embed URLs: /embed/job_board?for=<token> or /embed/job_app?for=<token>&token=<id>
        if parts and parts[0] == "embed" and "for" in qs:
            token = qs["for"][0].strip()
            job_id = qs["token"][0].strip() if "token" in qs else None
            if token:
                canonical = (
                    f"https://boards.greenhouse.io/{token}/jobs/{job_id}"
                    if job_id
                    else f"https://boards.greenhouse.io/{token}"
                )
                return DetectedATS(
                    ats_provider=ATSProvider.GREENHOUSE,
                    board_token=token,
                    external_job_id=job_id,
                    canonical_url=canonical,
                )
        if parts and parts[0] not in _IGNORED_SEGMENTS:
            token = parts[0]
            job_id = parts[2] if len(parts) >= 3 and parts[1] == "jobs" else None
            canonical = (
                f"https://{host}/{token}/jobs/{job_id}"
                if job_id
                else f"https://{host}/{token}"
            )
            return DetectedATS(
                ats_provider=ATSProvider.GREENHOUSE,
                board_token=token,
                external_job_id=job_id,
                canonical_url=canonical,
            )

    if host in _LEVER_HOSTS and parts and parts[0] not in _IGNORED_SEGMENTS:
        token = parts[0]
        job_id = parts[1] if len(parts) >= 2 and parts[1] not in _IGNORED_SEGMENTS else None
        canonical = f"https://{host}/{token}/{job_id}" if job_id else f"https://{host}/{token}"
        return DetectedATS(
            ats_provider=ATSProvider.LEVER,
            board_token=token,
            external_job_id=job_id,
            canonical_url=canonical,
        )

    if host in _ASHBY_HOSTS and parts and parts[0] not in _IGNORED_SEGMENTS:
        token = parts[0]
        job_id = parts[1] if len(parts) >= 2 and parts[1] not in _IGNORED_SEGMENTS else None
        canonical = f"https://{host}/{token}/{job_id}" if job_id else f"https://{host}/{token}"
        return DetectedATS(
            ats_provider=ATSProvider.ASHBY,
            board_token=token,
            external_job_id=job_id,
            canonical_url=canonical,
        )

    return None


def detect_ats_from_html(html_text: str) -> DetectedATS | None:
    """Scan an HTML careers page for embedded Greenhouse, Lever, or Ashby links."""
    embed_match = _EMBED_JS_PATTERN.search(html_text)
    if embed_match:
        token = embed_match.group(1)
        return DetectedATS(
            ats_provider=ATSProvider.GREENHOUSE,
            board_token=token,
            canonical_url=f"https://boards.greenhouse.io/{token}",
        )

    soup = BeautifulSoup(html_text, "html.parser")
    candidates: list[str] = []
    for tag, attr in (("iframe", "src"), ("a", "href"), ("script", "src")):
        for el in soup.find_all(tag):
            val = el.get(attr)
            if isinstance(val, str) and val.startswith(("http://", "https://")):
                candidates.append(val)

    for candidate in candidates:
        detected = detect_ats_from_url(candidate)
        if detected is not None:
            return detected
    return None
