"""Jobicy public remote developer jobs JSON collector."""

import httpx

from doot.collectors.ats.common import fetch_ats_json, html_to_text
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.job import RawJob


class JobicyCollector:
    """Fetches remote engineering jobs from `jobicy.com/api/v2/remote-jobs`."""

    source_name = "jobicy"

    def __init__(
        self,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
        *,
        count: int = 50,
    ) -> None:
        self.client = client
        self.rate_limiter = rate_limiter
        self.count = count

    async def fetch(self) -> list[RawJob]:
        """Fetch developer remote jobs from Jobicy public JSON API."""
        url = f"https://jobicy.com/api/v2/remote-jobs?count={min(self.count, 50)}&industry=dev"
        payload = await fetch_ats_json(self.client, self.rate_limiter, url)
        items = payload.get("jobs", []) if isinstance(payload, dict) else []

        raw_jobs: list[RawJob] = []
        for item in items:
            job_id = str(item.get("id") or "").strip()
            job_url = str(item.get("url") or "").strip()
            if not job_id or not job_url:
                continue

            geo = str(item.get("jobGeo") or "Anywhere").strip()
            location_raw = f"Remote ({geo})" if "remote" not in geo.lower() else geo

            min_sal = item.get("salaryMin")
            max_sal = item.get("salaryMax")
            curr = str(item.get("salaryCurrency") or "USD")
            salary_raw = (
                f"{curr} {min_sal}-{max_sal}"
                if (min_sal or max_sal)
                else None
            )

            raw_jobs.append(
                RawJob(
                    source=self.source_name,
                    external_job_id=job_id,
                    title=str(item.get("jobTitle") or "").strip(),
                    company_name=str(item.get("companyName") or "Unknown").strip(),
                    canonical_url=job_url,
                    apply_url=job_url,
                    location_raw=location_raw,
                    jd_text=html_to_text(str(item.get("jobDescription") or "")),
                    salary_raw=salary_raw,
                    alert_only=True,
                )
            )
        return raw_jobs
