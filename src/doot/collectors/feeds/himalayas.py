"""Himalayas.app public remote jobs JSON collector."""

import httpx

from doot.collectors.ats.common import fetch_ats_json, html_to_text
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.job import RawJob


class HimalayasCollector:
    """Fetches remote software engineering jobs from `himalayas.app/jobs/api`."""

    source_name = "himalayas"

    def __init__(
        self,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
        *,
        limit: int = 100,
    ) -> None:
        self.client = client
        self.rate_limiter = rate_limiter
        self.limit = limit

    async def fetch(self) -> list[RawJob]:
        """Fetch paginated remote postings from Himalayas public API."""
        raw_jobs: list[RawJob] = []
        offset = 0
        page_size = 20

        while len(raw_jobs) < self.limit:
            url = f"https://himalayas.app/jobs/api?limit={page_size}&offset={offset}"
            payload = await fetch_ats_json(self.client, self.rate_limiter, url)
            items = payload.get("jobs", []) if isinstance(payload, dict) else []
            if not items:
                break

            for item in items:
                guid = str(item.get("guid") or item.get("applicationLink") or "").strip()
                app_link = str(item.get("applicationLink") or guid).strip()
                if not guid or not app_link:
                    continue

                ext_id = guid.rstrip("/").rsplit("/", 1)[-1]
                restrictions = item.get("locationRestrictions") or []
                if isinstance(restrictions, list) and restrictions:
                    loc = f"Remote ({', '.join(str(r) for r in restrictions)})"
                else:
                    loc = "Remote (Worldwide)"

                min_sal = item.get("minSalary")
                max_sal = item.get("maxSalary")
                curr = str(item.get("currency") or "USD")
                salary_raw = (
                    f"{curr} {min_sal}-{max_sal}"
                    if (min_sal is not None or max_sal is not None)
                    else None
                )

                raw_jobs.append(
                    RawJob(
                        source=self.source_name,
                        external_job_id=ext_id,
                        title=str(item.get("title") or "").strip(),
                        company_name=str(item.get("companyName") or "Unknown").strip(),
                        canonical_url=guid,
                        apply_url=app_link,
                        location_raw=loc,
                        jd_text=html_to_text(str(item.get("description") or "")),
                        salary_raw=salary_raw,
                        alert_only=True,
                    )
                )
                if len(raw_jobs) >= self.limit:
                    break

            if len(items) < page_size:
                break
            offset += len(items)

        return raw_jobs
