"""Lever public postings JSON API collector."""

import httpx

from doot.collectors.ats.common import fetch_ats_json, html_to_text
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.enums import ATSProvider
from doot.models.job import CompanyRecord, RawJob


class LeverCollector:
    """Fetches jobs from `api.lever.co/v0/postings/{token}?mode=json`."""

    source_name = "lever"

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
        """Fetch and map Lever postings into RawJob records."""
        url = f"https://api.lever.co/v0/postings/{self.company.board_token}?mode=json"
        payload = await fetch_ats_json(self.client, self.rate_limiter, url)
        if not isinstance(payload, list):
            return []

        raw_jobs: list[RawJob] = []
        for item in payload:
            job_id = str(item["id"])
            hosted_url = str(item.get("hostedUrl") or "")
            apply_url = str(item.get("applyUrl") or (f"{hosted_url}/apply" if hosted_url else ""))
            categories = item.get("categories") or {}
            location_parts = [
                str(categories.get("location") or "").strip(),
                str(item.get("workplaceType") or "").strip(),
            ]
            location_raw = " | ".join(part for part in location_parts if part)

            plain_desc = item.get("descriptionPlain") or html_to_text(
                str(item.get("description") or "")
            )
            desc_sections = [str(plain_desc)]
            for section in item.get("lists") or []:
                header = str(section.get("text") or "").strip()
                body = html_to_text(str(section.get("content") or ""))
                if header or body:
                    desc_sections.append(f"{header}\n{body}".strip())
            jd_text = "\n\n".join(part for part in desc_sections if part)

            raw_jobs.append(
                RawJob(
                    source=self.source_name,
                    external_job_id=job_id,
                    title=str(item.get("text") or "").strip(),
                    company_name=self.company.name,
                    canonical_url=hosted_url,
                    apply_url=apply_url,
                    location_raw=location_raw,
                    jd_text=jd_text,
                    ats_provider=ATSProvider.LEVER,
                    board_token=self.company.board_token,
                    alert_only=False,
                )
            )
        return raw_jobs
