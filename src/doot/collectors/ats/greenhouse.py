"""Greenhouse public job-board JSON API collector."""

import httpx

from doot.collectors.ats.common import fetch_ats_json, html_to_text
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.enums import ATSProvider
from doot.models.job import CompanyRecord, RawJob


class GreenhouseCollector:
    """Fetches jobs from `boards-api.greenhouse.io/v1/boards/{token}/jobs`."""

    source_name = "greenhouse"

    def __init__(
        self,
        company: CompanyRecord,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
    ) -> None:
        self.company = company
        self.client = client
        self.rate_limiter = rate_limiter

    async def fetch(self) -> list[RawJob]:
        """Fetch and map Greenhouse board postings into RawJob records."""
        url = (
            f"https://boards-api.greenhouse.io/v1/boards/"
            f"{self.company.board_token}/jobs?content=true"
        )
        payload = await fetch_ats_json(self.client, self.rate_limiter, url)
        raw_jobs: list[RawJob] = []
        for item in payload.get("jobs", []):
            job_id = str(item["id"])
            absolute_url = str(item.get("absolute_url") or "")
            location_obj = item.get("location") or {}
            location_raw = str(location_obj.get("name") or "")
            jd_text = html_to_text(str(item.get("content") or ""))
            raw_jobs.append(
                RawJob(
                    source=self.source_name,
                    external_job_id=job_id,
                    title=str(item.get("title") or "").strip(),
                    company_name=self.company.name,
                    canonical_url=absolute_url,
                    apply_url=f"{absolute_url}#app" if absolute_url else "",
                    location_raw=location_raw,
                    jd_text=jd_text,
                    ats_provider=ATSProvider.GREENHOUSE,
                    board_token=self.company.board_token,
                    alert_only=False,
                )
            )
        return raw_jobs
