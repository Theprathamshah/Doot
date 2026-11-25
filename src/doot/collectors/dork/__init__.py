"""Search-dork discovery package."""

from doot.collectors.dork.discoverer import (
    DorkDiscoverer,
    DorkDiscoveryResult,
    DorkPack,
    DorkQuerySpec,
    load_dork_pack,
)
from doot.collectors.dork.search_providers import (
    BraveSearchProvider,
    DuckDuckGoSearchProvider,
    FallbackSearchProvider,
    SerperSearchProvider,
    TavilySearchProvider,
    build_search_provider,
)

__all__ = [
    "BraveSearchProvider",
    "DorkDiscoverer",
    "DorkDiscoveryResult",
    "DorkPack",
    "DorkQuerySpec",
    "DuckDuckGoSearchProvider",
    "FallbackSearchProvider",
    "SerperSearchProvider",
    "TavilySearchProvider",
    "build_search_provider",
    "load_dork_pack",
]
