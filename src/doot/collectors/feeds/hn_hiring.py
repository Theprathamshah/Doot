"""Hacker News 'Who is hiring?' Algolia API collector."""

import html
import re

import httpx
from bs4 import BeautifulSoup

from doot.collectors.ats.common import fetch_ats_json, html_to_text
from doot.collectors.ats.detector import detect_ats_from_url
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.job import RawJob

_URL_RE = re.compile(r"https?://[^\s<>\"')]+")


def _extract_links_from_html(raw_html: str) -> list[str]:
    """Extract full href URLs before text truncation (HN shortens visible link text)."""
    soup = BeautifulSoup(html.unescape(raw_html), "html.parser")
    hrefs: list[str] = []
    for tag in soup.find_all("a"):
        href = tag.get("href")
        if isinstance(href, str) and href.startswith(("http://", "https://")):
            hrefs.append(href.strip())
    return hrefs


class HNHiringCollector:
    """Fetches startup job comments from the latest HN 'Ask HN: Who is hiring?' thread."""

    source_name = "hn_hiring"

    def __init__(
        self,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
        *,
        limit: int = 80,
    ) -> None:
        self.client = client
        self.rate_limiter = rate_limiter
        self.limit = limit

    async def fetch(self) -> list[RawJob]:
        """Fetch latest 'Who is hiring?' thread via Algolia API and parse job comments."""
        search_url = (
            "https://hn.algolia.com/api/v1/search_by_date"
            "?query=%22Who%20is%20hiring%22&tags=story,author_whoishiring&hitsPerPage=1"
        )
        search_payload = await fetch_ats_json(self.client, self.rate_limiter, search_url)
        hits = search_payload.get("hits", []) if isinstance(search_payload, dict) else []
        if not hits:
            return []

        story_id = str(hits[0].get("objectID") or "").strip()
        if not story_id:
            return []

        item_url = f"https://hn.algolia.com/api/v1/items/{story_id}"
        story_payload = await fetch_ats_json(self.client, self.rate_limiter, item_url)
        children = story_payload.get("children", []) if isinstance(story_payload, dict) else []

        raw_jobs: list[RawJob] = []
        for comment in children:
            cid = str(comment.get("id") or "").strip()
            text_html = str(comment.get("text") or "").strip()
            if not cid or not text_html:
                continue

            plain = html_to_text(text_html)
            if not plain:
                continue

            first_line = plain.splitlines()[0]
            parts = [p.strip() for p in first_line.split("|") if p.strip()]
            company = parts[0][:80] if parts else "HN Startup"
            title = parts[1][:120] if len(parts) >= 2 else "Software Engineer (HN)"
            location_raw = parts[2][:120] if len(parts) >= 3 else (
                "Remote" if "remote" in first_line.lower() else ""
            )

            urls = _extract_links_from_html(text_html) or _URL_RE.findall(plain)
            hn_url = f"https://news.ycombinator.com/item?id={cid}"
            apply_url = urls[0].rstrip(".,;:") if urls else hn_url

            detected = None
            for u in urls:
                detected = detect_ats_from_url(u)
                if detected is not None:
                    apply_url = detected.canonical_url or u
                    break

            raw_jobs.append(
                RawJob(
                    source=self.source_name,
                    external_job_id=cid,
                    title=title,
                    company_name=company,
                    canonical_url=hn_url,
                    apply_url=apply_url,
                    location_raw=location_raw,
                    jd_text=plain,
                    ats_provider=detected.ats_provider if detected else None,
                    board_token=detected.board_token if detected else None,
                    alert_only=detected is None,
                )
            )
            if len(raw_jobs) >= self.limit:
                break

        return raw_jobs
