"""Remotive public remote software-dev JSON collector."""

import httpx

from doot.collectors.ats.common import fetch_ats_json, html_to_text
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.job import RawJob


class RemotiveCollector:
    """Fetches remote software jobs from `remotive.com/api/remote-jobs`."""

    source_name = "remotive"

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
        """Fetch software-dev remote postings from Remotive public API."""
        url = f"https://remotive.com/api/remote-jobs?category=software-dev&limit={self.limit}"
        payload = await fetch_ats_json(self.client, self.rate_limiter, url)
        items = payload.get("jobs", []) if isinstance(payload, dict) else []

        raw_jobs: list[RawJob] = []
        for item in items[: self.limit]:
            job_id = str(item.get("id") or "").strip()
            job_url = str(item.get("url") or "").strip()
            if not job_id or not job_url:
                continue

            req_loc = str(item.get("candidate_required_location") or "Worldwide").strip()
            location_raw = f"Remote ({req_loc})" if "remote" not in req_loc.lower() else req_loc
            salary_raw = str(item.get("salary") or "").strip() or None

            raw_jobs.append(
                RawJob(
                    source=self.source_name,
                    external_job_id=job_id,
                    title=str(item.get("title") or "").strip(),
                    company_name=str(item.get("company_name") or "Unknown").strip(),
                    canonical_url=job_url,
                    apply_url=job_url,
                    location_raw=location_raw,
                    jd_text=html_to_text(str(item.get("description") or "")),
                    salary_raw=salary_raw,
                    alert_only=True,
                )
            )
        return raw_jobs
