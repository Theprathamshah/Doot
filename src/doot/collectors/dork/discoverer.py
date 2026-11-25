"""Dork-to-Crawl discoverer: structured query packs, ATS extraction, and post-filters."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from doot.collectors.ats.detector import detect_ats_from_url
from doot.collectors.base import SearchHit, SearchProvider
from doot.db.repository import DootRepository
from doot.models.enums import CompanySource
from doot.models.job import RawJob

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
    ("gandhinagar", "Gandhinagar, India"),
    ("delhi", "Delhi NCR, India"),
)


class DorkQuerySpec(BaseModel):
    """Single dork query tagged with priority and category."""

    query: str
    priority: int = 1
    category: str = "general"


class DorkPack(BaseModel):
    """Structured search-dork configuration parsed from dork_queries.yaml."""

    freshness_days: int = 7
    max_results_per_query: int = 20
    suffix: str = ""
    reject_title_regex: str | None = None
    prefer_title_regex: str | None = None
    salary_signal_regex: str | None = None
    queries: list[DorkQuerySpec] = Field(default_factory=list)


def _format_company_or_group(names: list[str]) -> str:
    quoted = [f'"{n.strip()}"' for n in names if str(n).strip()]
    return f"({' OR '.join(quoted)})"


def load_dork_pack(path: Path) -> DorkPack:
    """Load either legacy flat `queries:` YAML or structured priority/global dork YAML."""
    if not path.exists():
        return DorkPack()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return DorkPack()

    global_cfg = data.get("global") or {}
    pack = DorkPack(
        freshness_days=int(global_cfg.get("freshness_days", 7)),
        max_results_per_query=int(global_cfg.get("max_results_per_query", 20)),
        suffix=str(global_cfg.get("suffix") or "").strip(),
        reject_title_regex=global_cfg.get("reject_title_regex"),
        prefer_title_regex=global_cfg.get("prefer_title_regex"),
        salary_signal_regex=global_cfg.get("salary_signal_regex"),
    )

    specs: list[DorkQuerySpec] = []
    # 1) Legacy top-level `queries: [...]`
    if isinstance(data.get("queries"), list):
        for q in data["queries"]:
            if str(q).strip():
                specs.append(DorkQuerySpec(query=str(q).strip(), priority=1, category="default"))

    # 2) Structured named sections (`ats_remote_india`, `startup_boards`, `target_companies`, ...)
    for key, section in data.items():
        if key in ("global", "queries") or not isinstance(section, dict):
            continue
        priority = int(section.get("priority", 1))
        for q in section.get("queries") or []:
            if str(q).strip():
                specs.append(
                    DorkQuerySpec(query=str(q).strip(), priority=priority, category=str(key))
                )
        chunks = section.get("company_chunks") or []
        templates = section.get("templates") or []
        for chunk in chunks:
            if not isinstance(chunk, list) or not chunk:
                continue
            companies_expr = _format_company_or_group([str(c) for c in chunk])
            for tmpl in templates:
                q_str = str(tmpl).replace("{companies}", companies_expr).strip()
                if q_str:
                    specs.append(DorkQuerySpec(query=q_str, priority=priority, category=str(key)))

    pack.queries = specs
    return pack


def _infer_company_name_from_token(token: str) -> str:
    cleaned = re.sub(r"[-_]+", " ", token).strip()
    return cleaned.title() if cleaned else token


def _infer_company_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    host = host.removeprefix("www.")
    return host.split(".")[0].title() if host else "Unknown"


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
    hits_rejected_by_title: int = 0
    new_companies_added: list[int] = field(default_factory=list)
    alert_only_jobs: list[RawJob] = field(default_factory=list)


class DorkDiscoverer:
    """Runs dork queries, registers discovered ATS boards, and emits non-ATS alert jobs."""

    def __init__(self, provider: SearchProvider, repo: DootRepository) -> None:
        self.provider = provider
        self.repo = repo

    async def discover(
        self,
        queries: list[str] | DorkPack,
        *,
        days: int = 7,
        max_results: int = 20,
        max_priority: int = 3,
    ) -> DorkDiscoveryResult:
        """Execute search dorks, apply optional title/salary post-filters, and register boards."""
        if isinstance(queries, list):
            pack = DorkPack(
                freshness_days=days,
                max_results_per_query=max_results,
                queries=[DorkQuerySpec(query=q, priority=1) for q in queries],
            )
        else:
            pack = queries

        reject_re = (
            re.compile(pack.reject_title_regex) if pack.reject_title_regex else None
        )
        salary_re = (
            re.compile(pack.salary_signal_regex) if pack.salary_signal_regex else None
        )

        result = DorkDiscoveryResult()
        seen_urls: set[str] = set()
        effective_days = pack.freshness_days or days
        effective_max = pack.max_results_per_query or max_results

        for spec in pack.queries:
            if spec.priority > max_priority:
                continue

            full_query = f"{spec.query} {pack.suffix}".strip() if pack.suffix else spec.query
            # DDG struggles with long boolean queries + negative terms; keep raw query for ddg
            if self.provider.provider_name == "ddg":
                full_query = spec.query

            result.queries_executed += 1
            hits: list[SearchHit] = await self.provider.search(
                full_query, days=effective_days, max_results=effective_max
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

                inferred_title, inferred_company = _infer_title_and_company(hit)
                if reject_re is not None and reject_re.search(inferred_title):
                    result.hits_rejected_by_title += 1
                    continue

                salary_raw: str | None = None
                if salary_re is not None:
                    sal_match = salary_re.search(f"{hit.title} {hit.snippet}")
                    if sal_match:
                        salary_raw = sal_match.group(0)

                ext_id = hashlib.sha256(hit.url.encode("utf-8")).hexdigest()[:24]
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
                        salary_raw=salary_raw,
                        alert_only=True,
                    )
                )

        return result
