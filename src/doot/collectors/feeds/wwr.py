"""WeWorkRemotely public programming RSS feed collector."""

import hashlib
import xml.etree.ElementTree as ET

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from doot.collectors.ats.common import html_to_text
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.job import RawJob

_WWR_RSS_URL = "https://weworkremotely.com/categories/remote-programming-jobs.rss"


class WeWorkRemotelyCollector:
    """Fetches remote programming postings from WeWorkRemotely's public RSS feed."""

    source_name = "weworkremotely"

    def __init__(
        self,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
        *,
        limit: int = 80,
    ) -> None:
        self.client = client
        self.rate_limiter = rate_limiter
        self.limit = limit

    async def fetch(self) -> list[RawJob]:
        """Fetch and parse WeWorkRemotely programming RSS items."""

        @retry(
            retry=retry_if_exception_type(httpx.TransportError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            reraise=True,
        )
        async def _get_xml() -> str:
            await self.rate_limiter.acquire(_WWR_RSS_URL)
            resp = await self.client.get(
                _WWR_RSS_URL,
                headers={"User-Agent": "DootJobTriage/0.1"},
                timeout=20.0,
            )
            resp.raise_for_status()
            return resp.text

        xml_text = await _get_xml()
        root = ET.fromstring(xml_text)
        raw_jobs: list[RawJob] = []

        for item in root.findall("./channel/item")[: self.limit]:
            raw_title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            region = (item.findtext("region") or "Anywhere in the World").strip()
            desc_html = item.findtext("description") or ""
            if not raw_title or not link:
                continue

            if ":" in raw_title:
                company_name, role_title = [p.strip() for p in raw_title.split(":", 1)]
            else:
                company_name, role_title = "WWR Company", raw_title

            ext_id = hashlib.sha256(link.encode("utf-8")).hexdigest()[:24]
            location_raw = f"Remote ({region})" if "remote" not in region.lower() else region

            raw_jobs.append(
                RawJob(
                    source=self.source_name,
                    external_job_id=ext_id,
                    title=role_title,
                    company_name=company_name,
                    canonical_url=link,
                    apply_url=link,
                    location_raw=location_raw,
                    jd_text=html_to_text(desc_html),
                    alert_only=True,
                )
            )
        return raw_jobs
