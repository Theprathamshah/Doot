"""Ashby public posting JSON API collector."""

import httpx

from doot.collectors.ats.common import fetch_ats_json, html_to_text
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.enums import ATSProvider
from doot.models.job import CompanyRecord, RawJob


class AshbyCollector:
    """Fetches jobs from `api.ashbyhq.com/posting-api/job-board/{token}`."""

    source_name = "ashby"

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
        """Fetch and map Ashby postings into RawJob records."""
        url = f"https://api.ashbyhq.com/posting-api/job-board/{self.company.board_token}"
        payload = await fetch_ats_json(self.client, self.rate_limiter, url)
        raw_jobs: list[RawJob] = []
        for item in payload.get("jobs", []):
            job_id = str(item["id"])
            job_url = str(item.get("jobUrl") or "")
            apply_url = str(item.get("applyUrl") or (f"{job_url}/application" if job_url else ""))
            location_raw = str(item.get("location") or "")
            if item.get("isRemote") and "remote" not in location_raw.lower():
                location_raw = f"{location_raw} (Remote)".strip()

            jd_text = str(item.get("descriptionPlain") or "").strip()
            if not jd_text:
                jd_text = html_to_text(str(item.get("descriptionHtml") or ""))

            compensation = item.get("compensation") or {}
            salary_raw = compensation.get("compensationTierSummary")

            raw_jobs.append(
                RawJob(
                    source=self.source_name,
                    external_job_id=job_id,
                    title=str(item.get("title") or "").strip(),
                    company_name=self.company.name,
                    canonical_url=job_url,
                    apply_url=apply_url,
                    location_raw=location_raw,
                    jd_text=jd_text,
                    salary_raw=str(salary_raw) if salary_raw else None,
                    ats_provider=ATSProvider.ASHBY,
                    board_token=self.company.board_token,
                    alert_only=False,
                )
            )
        return raw_jobs
