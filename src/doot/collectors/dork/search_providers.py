"""SearchProvider implementations for Serper, Brave Search, and Tavily."""

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from doot.collectors.base import SearchHit, SearchProvider
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.config import DootSettings

_RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})


class TransientSearchError(RuntimeError):
    """Raised when a search API returns a retryable 429/5xx response."""


async def _post_or_get_json(
    client: httpx.AsyncClient,
    rate_limiter: DomainRateLimiter,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    @retry(
        retry=retry_if_exception_type((TransientSearchError, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def _request() -> Any:
        await rate_limiter.acquire(url)
        resp = await client.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=20.0,
        )
        if resp.status_code in _RETRYABLE_CODES:
            raise TransientSearchError(f"Search API {resp.status_code} at {url}")
        resp.raise_for_status()
        return resp.json()

    return await _request()


def _days_to_google_tbs(days: int) -> str:
    if days <= 1:
        return "qdr:d"
    if days <= 7:
        return "qdr:w"
    return "qdr:m"


def _days_to_brave_freshness(days: int) -> str:
    if days <= 1:
        return "pd"
    if days <= 7:
        return "pw"
    return "pm"


class SerperSearchProvider:
    """Google search results via Serper.dev JSON API."""

    provider_name = "serper"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
    ) -> None:
        self.api_key = api_key
        self.client = client
        self.rate_limiter = rate_limiter

    async def search(
        self,
        query: str,
        *,
        days: int = 7,
        max_results: int = 20,
    ) -> list[SearchHit]:
        payload = await _post_or_get_json(
            self.client,
            self.rate_limiter,
            "POST",
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json_body={
                "q": query,
                "num": min(max_results, 100),
                "tbs": _days_to_google_tbs(days),
            },
        )
        hits: list[SearchHit] = []
        for item in payload.get("organic", [])[:max_results]:
            link = str(item.get("link") or "").strip()
            if not link:
                continue
            hits.append(
                SearchHit(
                    url=link,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("snippet") or ""),
                    provider=self.provider_name,
                )
            )
        return hits


class BraveSearchProvider:
    """Web search results via Brave Search JSON API."""

    provider_name = "brave"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
    ) -> None:
        self.api_key = api_key
        self.client = client
        self.rate_limiter = rate_limiter

    async def search(
        self,
        query: str,
        *,
        days: int = 7,
        max_results: int = 20,
    ) -> list[SearchHit]:
        payload = await _post_or_get_json(
            self.client,
            self.rate_limiter,
            "GET",
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            params={
                "q": query,
                "count": min(max_results, 20),
                "freshness": _days_to_brave_freshness(days),
            },
        )
        results = (payload.get("web") or {}).get("results", [])
        hits: list[SearchHit] = []
        for item in results[:max_results]:
            link = str(item.get("url") or "").strip()
            if not link:
                continue
            hits.append(
                SearchHit(
                    url=link,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("description") or ""),
                    provider=self.provider_name,
                )
            )
        return hits


class TavilySearchProvider:
    """Web search results via Tavily Search API."""

    provider_name = "tavily"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
    ) -> None:
        self.api_key = api_key
        self.client = client
        self.rate_limiter = rate_limiter

    async def search(
        self,
        query: str,
        *,
        days: int = 7,
        max_results: int = 20,
    ) -> list[SearchHit]:
        payload = await _post_or_get_json(
            self.client,
            self.rate_limiter,
            "POST",
            "https://api.tavily.com/search",
            json_body={
                "api_key": self.api_key,
                "query": query,
                "max_results": min(max_results, 20),
                "days": days,
                "search_depth": "basic",
            },
        )
        hits: list[SearchHit] = []
        for item in payload.get("results", [])[:max_results]:
            link = str(item.get("url") or "").strip()
            if not link:
                continue
            hits.append(
                SearchHit(
                    url=link,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("content") or ""),
                    provider=self.provider_name,
                )
            )
        return hits


def build_search_provider(
    settings: DootSettings,
    client: httpx.AsyncClient,
    rate_limiter: DomainRateLimiter,
) -> SearchProvider | None:
    """Construct the configured SearchProvider if an API key is available."""
    provider = settings.search_provider.strip().lower()
    if provider == "serper" and settings.serper_api_key.get_secret_value():
        return SerperSearchProvider(
            settings.serper_api_key.get_secret_value(), client, rate_limiter
        )
    if provider == "brave" and settings.brave_api_key.get_secret_value():
        return BraveSearchProvider(
            settings.brave_api_key.get_secret_value(), client, rate_limiter
        )
    if provider == "tavily" and settings.tavily_api_key.get_secret_value():
        return TavilySearchProvider(
            settings.tavily_api_key.get_secret_value(), client, rate_limiter
        )
    return None
