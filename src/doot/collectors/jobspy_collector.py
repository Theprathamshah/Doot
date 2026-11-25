"""Logged-out secondary job board collector wrapping synchronous python-jobspy."""

import asyncio
import hashlib
from collections.abc import Callable
from typing import Any

from loguru import logger

from doot.collectors.ats.detector import detect_ats_from_url
from doot.models.job import RawJob

ScrapeCallable = Callable[..., Any]


def _default_scrape_jobs(**kwargs: Any) -> Any:
    try:
        from jobspy import scrape_jobs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "python-jobspy is not installed. Install optional extra with `uv sync --extra jobspy`."
        ) from exc
    return scrape_jobs(**kwargs)


def _records_from_result(raw_result: Any) -> list[dict[str, Any]]:
    if raw_result is None:
        return []
    if isinstance(raw_result, list):
        return [dict(item) for item in raw_result]
    to_dict = getattr(raw_result, "to_dict", None)
    if callable(to_dict):
        records = to_dict(orient="records")
        return [dict(item) for item in records]
    return []


def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    text = str(val).strip()
    return "" if text.lower() == "nan" else text


class JobSpyCollector:
    """Async wrapper around logged-out python-jobspy scraping."""

    source_name = "jobspy"

    def __init__(
        self,
        *,
        sites: list[str],
        search_terms: list[str],
        location: str = "India",
        hours_old: int = 24,
        results_wanted: int = 20,
        timeout_seconds: float = 45.0,
        scrape_fn: ScrapeCallable | None = None,
    ) -> None:
        self.sites = sites
        self.search_terms = search_terms
        self.location = location
        self.hours_old = hours_old
        self.results_wanted = results_wanted
        self.timeout_seconds = timeout_seconds
        self._scrape_fn = scrape_fn or _default_scrape_jobs

    async def fetch(self) -> list[RawJob]:
        """Run logged-out JobSpy queries in worker threads with a strict timeout."""
        collected: list[RawJob] = []
        seen_ids: set[tuple[str, str]] = set()

        for term in self.search_terms:
            try:
                raw_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._scrape_fn,
                        site_name=self.sites,
                        search_term=term,
                        location=self.location,
                        results_wanted=self.results_wanted,
                        hours_old=self.hours_old,
                        country_indeed="India",
                        linkedin_fetch_description=True,
                    ),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                logger.warning(
                    "JobSpy timed out after {}s for term '{}'", self.timeout_seconds, term
                )
                continue

            for row in _records_from_result(raw_result):
                site = _clean_str(row.get("site")) or "unknown"
                job_url = _clean_str(row.get("job_url"))
                direct_url = _clean_str(row.get("job_url_direct"))
                target_url = direct_url or job_url
                if not target_url:
                    continue

                ext_id = _clean_str(row.get("id")) or hashlib.sha256(
                    target_url.encode("utf-8")
                ).hexdigest()[:20]
                source = f"jobspy_{site.lower()}"
                if (source, ext_id) in seen_ids:
                    continue
                seen_ids.add((source, ext_id))

                detected = detect_ats_from_url(direct_url or job_url)
                # Keep LinkedIn/Indeed/Naukri alert_only unless resolved to a supported ATS
                alert_only = detected is None
                resolved_apply_url = (
                    detected.canonical_url
                    if detected and detected.canonical_url
                    else target_url
                )

                min_amt = _clean_str(row.get("min_amount"))
                max_amt = _clean_str(row.get("max_amount"))
                currency = _clean_str(row.get("currency"))
                salary_raw = (
                    f"{currency} {min_amt}-{max_amt}".strip()
                    if (min_amt or max_amt)
                    else None
                )

                collected.append(
                    RawJob(
                        source=source,
                        external_job_id=ext_id,
                        title=_clean_str(row.get("title")) or "Untitled Role",
                        company_name=_clean_str(row.get("company")) or "Unknown",
                        canonical_url=job_url or target_url,
                        apply_url=resolved_apply_url,
                        location_raw=_clean_str(row.get("location")),
                        jd_text=_clean_str(row.get("description")),
                        salary_raw=salary_raw,
                        ats_provider=detected.ats_provider if detected else None,
                        board_token=detected.board_token if detected else None,
                        alert_only=alert_only,
                    )
                )

        return collected
