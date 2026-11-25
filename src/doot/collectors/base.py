"""Collector and search-provider Protocols."""

from typing import Protocol

from pydantic import BaseModel

from doot.models.job import RawJob


class SearchHit(BaseModel):
    """Normalized web search result returned by a SearchProvider."""

    url: str
    title: str = ""
    snippet: str = ""
    provider: str = ""


class SearchProvider(Protocol):
    """Interface for search-dorking providers (Serper, Brave, Tavily, etc.)."""

    provider_name: str

    async def search(
        self,
        query: str,
        *,
        days: int = 7,
        max_results: int = 20,
    ) -> list[SearchHit]:
        """Execute a search query with recency filtering."""
        ...


class Collector(Protocol):
    """Interface for all job discovery collectors."""

    source_name: str

    async def fetch(self) -> list[RawJob]:
        """Fetch raw job postings from the underlying source."""
        ...
