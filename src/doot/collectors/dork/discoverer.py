"""Dork-to-Crawl discoverer: extracts ATS boards and alert-only jobs from search hits."""

import hashlib
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from loguru import logger

from doot.collectors.ats.detector import detect_ats_from_url
from doot.collectors.base import SearchHit, SearchProvider
from doot.db.repository import DootRepository
from doot.models.enums import CompanySource
from doot.models.job import RawJob


def _infer_company_name_from_token(token: str) -> str:
    cleaned = re.sub(r"[-_]+", " ", token).strip()
    return cleaned.title() if cleaned else token


def _infer_company_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    host = host.removeprefix("www.")
    return host.split(".")[0].title() if host else "Unknown"


_INDIA_CITY_TOKENS: tuple[tuple[str, str], ...] = (
    ("bengaluru", "Bengaluru, India"),
    ("bangalore", "Bengaluru, India"),
    ("hyderabad", "Hyderabad, India"),
    ("pune", "Pune, India"),
    ("gurugram", "Gurugram, India"),
    ("gurgaon", "Gurugram, India"),
    ("noida", "Noida, India"),
    ("mumbai", "Mumbai, India"),
    ("chennai", "Chennai, India"),
    ("ahmedabad", "Ahmedabad, India"),
    ("delhi", "Delhi NCR, India"),
)


def _infer_location_from_hit(hit: SearchHit) -> str:
    combined = f"{hit.url} {hit.title} {hit.snippet}".lower()
    is_remote = "remote" in combined
    for token, formatted in _INDIA_CITY_TOKENS:
        if token in combined:
            return f"{formatted} (Remote/Hybrid)" if is_remote else formatted
    if is_remote and "india" in combined:
        return "Remote (India)"
    if is_remote:
        return "Remote"
    if "india" in combined:
        return "India"
    return ""


def _infer_title_and_company(hit: SearchHit) -> tuple[str, str]:
    raw_title = hit.title.strip()
    # LinkedIn format: "Backend Engineer - Acme Corp | LinkedIn" or "Acme hiring Backend Engineer"
    cleaned = re.sub(r"\s*\|\s*(?:LinkedIn|Wellfound|AngelList).*$", "", raw_title, flags=re.I)
    if " hiring " in cleaned.lower():
        parts = re.split(r"\s+hiring\s+", cleaned, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            return parts[1].strip(), parts[0].strip()
    if " - " in cleaned:
        left, right = cleaned.rsplit(" - ", 1)
        if left and right and len(right) <= 50:
            return left.strip(), right.strip()
    if " at " in cleaned.lower():
        parts = re.split(r"\s+at\s+", cleaned, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return cleaned or hit.url, _infer_company_from_url(hit.url)


@dataclass(slots=True)
class DorkDiscoveryResult:
    """Output of a Dork-to-Crawl sweep."""

    queries_executed: int = 0
    hits_seen: int = 0
    new_companies_added: list[int] = field(default_factory=list)
    alert_only_jobs: list[RawJob] = field(default_factory=list)


class DorkDiscoverer:
    """Runs dork queries, registers discovered ATS boards, and emits non-ATS alert jobs."""

    def __init__(self, provider: SearchProvider, repo: DootRepository) -> None:
        self.provider = provider
        self.repo = repo

    async def discover(
        self,
        queries: list[str],
        *,
        days: int = 7,
        max_results: int = 20,
    ) -> DorkDiscoveryResult:
        """Execute search dorks and register newly discovered ATS boards."""
        result = DorkDiscoveryResult()
        seen_urls: set[str] = set()

        for query in queries:
            result.queries_executed += 1
            hits: list[SearchHit] = await self.provider.search(
                query, days=days, max_results=max_results
            )
            for hit in hits:
                result.hits_seen += 1
                if hit.url in seen_urls:
                    continue
                seen_urls.add(hit.url)

                detected = detect_ats_from_url(hit.url)
                if detected is not None:
                    existing = await self.repo.get_company_by_ats(
                        detected.ats_provider, detected.board_token
                    )
                    if existing is None:
                        inferred_name = _infer_company_name_from_token(detected.board_token)
                        company_id = await self.repo.upsert_company(
                            name=inferred_name,
                            normalized_name=inferred_name.lower(),
                            ats_provider=detected.ats_provider,
                            board_token=detected.board_token,
                            careers_url=detected.canonical_url,
                            source=CompanySource.DORK,
                        )
                        result.new_companies_added.append(company_id)
                        logger.info(
                            "Discovered ATS board via dork: {} ({})",
                            detected.board_token,
                            detected.ats_provider.value,
                        )
                    continue

                # Non-ATS hit (e.g., Wellfound, LinkedIn, or custom careers page) -> alert_only
                ext_id = hashlib.sha256(hit.url.encode("utf-8")).hexdigest()[:24]
                inferred_title, inferred_company = _infer_title_and_company(hit)
                inferred_loc = _infer_location_from_hit(hit)
                result.alert_only_jobs.append(
                    RawJob(
                        source="dork",
                        external_job_id=ext_id,
                        title=inferred_title,
                        company_name=inferred_company,
                        canonical_url=hit.url,
                        apply_url=hit.url,
                        location_raw=inferred_loc,
                        jd_text=hit.snippet,
                        alert_only=True,
                    )
                )

        return result
