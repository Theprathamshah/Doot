"""ATS collector registry and URL detector exports."""

import httpx

from doot.collectors.ats.ashby import AshbyCollector
from doot.collectors.ats.common import ATSNotFoundError
from doot.collectors.ats.detector import DetectedATS, detect_ats_from_html, detect_ats_from_url
from doot.collectors.ats.greenhouse import GreenhouseCollector
from doot.collectors.ats.lever import LeverCollector
from doot.collectors.base import Collector
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.enums import ATSProvider
from doot.models.job import CompanyRecord

_CollectorCls = type[GreenhouseCollector | LeverCollector | AshbyCollector]
_ATS_COLLECTOR_MAP: dict[ATSProvider, _CollectorCls] = {
    ATSProvider.GREENHOUSE: GreenhouseCollector,
    ATSProvider.LEVER: LeverCollector,
    ATSProvider.ASHBY: AshbyCollector,
}


def build_ats_collector(
    company: CompanyRecord,
    client: httpx.AsyncClient,
    rate_limiter: DomainRateLimiter,
) -> Collector | None:
    """Instantiate the appropriate ATS JSON collector for a company record."""
    cls = _ATS_COLLECTOR_MAP.get(company.ats_provider)
    if cls is None:
        return None
    return cls(company, client, rate_limiter)


__all__ = [
    "ATSNotFoundError",
    "AshbyCollector",
    "DetectedATS",
    "GreenhouseCollector",
    "LeverCollector",
    "build_ats_collector",
    "detect_ats_from_html",
    "detect_ats_from_url",
]
