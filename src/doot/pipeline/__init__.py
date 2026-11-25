"""Job normalization, filtering, scoring, and discovery orchestration."""

from doot.pipeline.discovery import (
    DiscoverySummary,
    add_company_from_target,
    run_discovery_pipeline,
    sync_watchlist_from_yaml,
)
from doot.pipeline.normalizer import (
    canonicalize_url,
    compute_dedupe_cluster_id,
    compute_jd_hash,
    normalize_raw_job,
    normalize_text,
)

__all__ = [
    "DiscoverySummary",
    "add_company_from_target",
    "canonicalize_url",
    "compute_dedupe_cluster_id",
    "compute_jd_hash",
    "normalize_raw_job",
    "normalize_text",
    "run_discovery_pipeline",
    "sync_watchlist_from_yaml",
]
