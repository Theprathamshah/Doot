"""Discovery pipeline orchestration (Watchlist sync, Dork-to-Crawl, ATS, JobSpy)."""

from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from loguru import logger

from doot.collectors.ats import (
    ATSNotFoundError,
    build_ats_collector,
    detect_ats_from_html,
    detect_ats_from_url,
)
from doot.collectors.base import Collector, SearchProvider
from doot.collectors.dork import DorkDiscoverer, DorkPack, load_dork_pack
from doot.collectors.feeds import (
    HimalayasCollector,
    HNHiringCollector,
    JobicyCollector,
    RedditHiringCollector,
    RemotiveCollector,
    WeWorkRemotelyCollector,
)
from doot.collectors.jobspy_collector import JobSpyCollector
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.config import DootSettings
from doot.db.repository import DootRepository
from doot.models.enums import ATSProvider, CompanySource, RunStatus, RunType
from doot.models.job import RawJob
from doot.pipeline.normalizer import normalize_raw_job, normalize_text


@dataclass(slots=True)
class DiscoverySummary:
    """Aggregated counts returned after a discovery run."""

    companies_crawled: int = 0
    companies_failed: int = 0
    companies_paused: int = 0
    new_boards_via_dork: int = 0
    jobs_created: int = 0
    jobs_updated: int = 0
    jobs_deduped_cross_source: int = 0
    errors: list[str] = field(default_factory=list)


def load_dork_queries(path: Path) -> list[str]:
    """Load flattened dork query strings from YAML if present."""
    return [spec.query for spec in load_dork_pack(path).queries]


async def sync_watchlist_from_yaml(repo: DootRepository, path: Path) -> int:
    """Import or update companies defined in a watchlist YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    count = 0
    for entry in data.get("companies", []):
        name = str(entry["name"]).strip()
        provider = ATSProvider(str(entry["ats_provider"]).strip().lower())
        token = str(entry["board_token"]).strip()
        careers_url = entry.get("careers_url")
        is_blocklisted = bool(entry.get("is_blocklisted", False))
        await repo.upsert_company(
            name=name,
            normalized_name=normalize_text(name),
            ats_provider=provider,
            board_token=token,
            careers_url=str(careers_url) if careers_url else None,
            source=CompanySource.MANUAL,
            is_blocklisted=is_blocklisted,
        )
        count += 1
    return count


async def add_company_from_target(
    repo: DootRepository,
    target: str,
    *,
    client: httpx.AsyncClient,
    name: str | None = None,
    ats_provider: ATSProvider | None = None,
    board_token: str | None = None,
) -> tuple[int, ATSProvider, str]:
    """Resolve an ATS provider/token from a URL, HTML page, or explicit flags and save it."""
    if ats_provider is not None and board_token:
        resolved_provider = ats_provider
        resolved_token = board_token.strip()
        careers_url = target if target.startswith(("http://", "https://")) else None
    else:
        detected = detect_ats_from_url(target)
        if detected is None and target.startswith(("http://", "https://")):
            resp = await client.get(target, follow_redirects=True, timeout=20.0)
            resp.raise_for_status()
            detected = detect_ats_from_url(str(resp.url)) or detect_ats_from_html(resp.text)
        if detected is None:
            raise ValueError(
                f"Could not detect Greenhouse/Lever/Ashby board from '{target}'. "
                "Provide --ats and --token explicitly."
            )
        resolved_provider = detected.ats_provider
        resolved_token = detected.board_token
        careers_url = detected.canonical_url or target

    display_name = name.strip() if name else resolved_token.replace("-", " ").title()
    company_id = await repo.upsert_company(
        name=display_name,
        normalized_name=normalize_text(display_name),
        ats_provider=resolved_provider,
        board_token=resolved_token,
        careers_url=careers_url,
        source=CompanySource.MANUAL,
    )
    return company_id, resolved_provider, resolved_token


async def _ingest_raw_jobs(
    repo: DootRepository,
    raw_jobs: list[RawJob],
    summary: DiscoverySummary,
    *,
    company_id: int | None = None,
) -> None:
    for raw in raw_jobs:
        posting = normalize_raw_job(raw, company_id=company_id)

        # Cross-source dedup: if JobSpy/Dork resolved to an ATS job already crawled, skip duplicate
        if (
            posting.ats_provider is not None
            and posting.board_token is not None
            and raw.source != posting.ats_provider.value
        ):
            detected = detect_ats_from_url(posting.canonical_url) or detect_ats_from_url(
                posting.apply_url
            )
            ats_job_id = detected.external_job_id if detected else None
            if ats_job_id:
                existing = await repo.get_job_by_ats_key(
                    posting.ats_provider, posting.board_token, ats_job_id
                )
                if existing is not None:
                    summary.jobs_deduped_cross_source += 1
                    continue

        _, created = await repo.upsert_job(posting, actor="discover")
        if created:
            summary.jobs_created += 1
        else:
            summary.jobs_updated += 1


async def run_discovery_pipeline(
    settings: DootSettings,
    repo: DootRepository,
    *,
    client: httpx.AsyncClient,
    rate_limiter: DomainRateLimiter,
    use_ats: bool = True,
    use_feeds: bool = False,
    use_dork: bool = False,
    use_jobspy: bool = False,
    dork_max_priority: int = 2,
    limit_companies: int | None = None,
    search_provider: SearchProvider | None = None,
    jobspy_collector: JobSpyCollector | None = None,
    feed_collectors: list[Collector] | None = None,
) -> DiscoverySummary:
    """Execute hybrid discovery (Dork -> Public Feeds -> ATS JSON -> JobSpy)."""
    run_id = await repo.start_run(RunType.DISCOVER)
    summary = DiscoverySummary()

    try:
        if use_dork and search_provider is not None:
            pack: DorkPack = load_dork_pack(settings.dork_queries_path)
            if not pack.queries and settings.dork_queries_path.with_name(
                "dork_queries.example.yaml"
            ).exists():
                pack = load_dork_pack(
                    settings.dork_queries_path.with_name("dork_queries.example.yaml")
                )
            if pack.queries:
                try:
                    dork_res = await DorkDiscoverer(search_provider, repo).discover(
                        pack,
                        days=settings.dork_days,
                        max_results=settings.dork_max_results,
                        max_priority=dork_max_priority,
                    )
                    summary.new_boards_via_dork = len(dork_res.new_companies_added)
                    await _ingest_raw_jobs(repo, dork_res.alert_only_jobs, summary)
                except Exception as exc:
                    summary.errors.append(f"dork ({search_provider.provider_name}): {exc}")
                    logger.warning("Dork discovery failed: {}", exc)

        if use_feeds:
            active_feeds: list[Collector] = (
                feed_collectors
                if feed_collectors is not None
                else [
                    HimalayasCollector(client, rate_limiter, limit=100),
                    RemotiveCollector(client, rate_limiter, limit=100),
                    JobicyCollector(client, rate_limiter, count=50),
                    HNHiringCollector(client, rate_limiter, limit=80),
                    WeWorkRemotelyCollector(client, rate_limiter, limit=80),
                    RedditHiringCollector(client, rate_limiter, limit_per_sub=20),
                ]
            )
            for feed in active_feeds:
                try:
                    feed_jobs = await feed.fetch()
                    # Register any ATS boards discovered in HN/feed links before ATS crawl
                    for rj in feed_jobs:
                        if rj.ats_provider is not None and rj.board_token:
                            existing_co = await repo.get_company_by_ats(
                                rj.ats_provider, rj.board_token
                            )
                            if existing_co is None:
                                await repo.upsert_company(
                                    name=rj.company_name[:80],
                                    normalized_name=normalize_text(rj.company_name[:80]),
                                    ats_provider=rj.ats_provider,
                                    board_token=rj.board_token,
                                    careers_url=rj.apply_url,
                                    source=CompanySource.SEED,
                                )
                    await _ingest_raw_jobs(repo, feed_jobs, summary)
                except Exception as exc:
                    summary.errors.append(f"feed ({feed.source_name}): {exc}")
                    logger.warning("Public feed {} failed: {}", feed.source_name, exc)

        if use_ats:
            companies = await repo.list_companies(active_only=True)
            if limit_companies is not None:
                companies = companies[:limit_companies]

            for company in companies:
                collector = build_ats_collector(company, client, rate_limiter)
                if collector is None:
                    continue
                try:
                    raw_jobs = await collector.fetch()
                except (ATSNotFoundError, httpx.HTTPError) as exc:
                    paused = await repo.record_company_crawl_failure(
                        company.id, max_failures=settings.ats_max_consecutive_failures
                    )
                    summary.companies_failed += 1
                    if paused:
                        summary.companies_paused += 1
                    summary.errors.append(f"{company.name} ({company.board_token}): {exc}")
                    logger.warning("ATS crawl failed for {}: {}", company.name, exc)
                    continue

                await repo.record_company_crawl_success(company.id)
                summary.companies_crawled += 1
                await _ingest_raw_jobs(repo, raw_jobs, summary, company_id=company.id)

        if use_jobspy:
            active_jobspy = jobspy_collector or JobSpyCollector(
                sites=settings.jobspy_sites,
                search_terms=settings.jobspy_search_terms,
                location=settings.jobspy_location,
                hours_old=settings.jobspy_hours_old,
                results_wanted=settings.jobspy_results_wanted,
                timeout_seconds=settings.jobspy_timeout_seconds,
            )
            jobspy_jobs = await active_jobspy.fetch()
            await _ingest_raw_jobs(repo, jobspy_jobs, summary)

        run_status = RunStatus.PARTIAL if summary.companies_failed > 0 else RunStatus.SUCCESS
        await repo.finish_run(
            run_id,
            status=run_status,
            stats={
                "companies_crawled": summary.companies_crawled,
                "companies_failed": summary.companies_failed,
                "companies_paused": summary.companies_paused,
                "new_boards_via_dork": summary.new_boards_via_dork,
                "jobs_created": summary.jobs_created,
                "jobs_updated": summary.jobs_updated,
                "jobs_deduped_cross_source": summary.jobs_deduped_cross_source,
            },
        )
        return summary
    except Exception as exc:
        await repo.finish_run(run_id, status=RunStatus.FAILED, error_message=str(exc))
        raise
