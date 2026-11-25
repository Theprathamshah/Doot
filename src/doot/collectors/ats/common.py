"""Shared HTTP fetching, retry logic, and HTML-to-text helpers for ATS collectors."""

import html
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from doot.collectors.rate_limiter import DomainRateLimiter

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_USER_AGENT = "DootJobTriage/0.1 (+https://github.com/local/doot)"


class ATSNotFoundError(RuntimeError):
    """Raised when an ATS board returns HTTP 404 (invalid or removed board token)."""


class TransientATSError(RuntimeError):
    """Raised for retryable HTTP 429/5xx responses."""


def html_to_text(raw_html: str) -> str:
    """Convert escaped or raw ATS HTML job descriptions into clean plain text."""
    if not raw_html:
        return ""
    unescaped = html.unescape(raw_html)
    soup = BeautifulSoup(unescaped, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        block.insert_before("\n")
        block.insert_after("\n")
    text = soup.get_text(separator=" ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


async def fetch_ats_json(
    client: httpx.AsyncClient,
    rate_limiter: DomainRateLimiter,
    url: str,
) -> Any:
    """Fetch JSON from an ATS public API endpoint with rate limiting and backoff."""

    @retry(
        retry=retry_if_exception_type((TransientATSError, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def _do_get() -> Any:
        await rate_limiter.acquire(url)
        resp = await client.get(url, headers={"User-Agent": _USER_AGENT}, timeout=20.0)
        if resp.status_code == 404:
            raise ATSNotFoundError(f"ATS board not found (404): {url}")
        if resp.status_code in _RETRYABLE_STATUS_CODES:
            raise TransientATSError(f"Retryable status {resp.status_code} from {url}")
        resp.raise_for_status()
        return resp.json()

    return await _do_get()
