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
from doot.pipeline.redactor import build_redacted_profile_summary, redact_pii
from doot.pipeline.rule_filter import RuleFilterDecision, evaluate_rule_filter
from doot.pipeline.scorer import (
    ScoringSummary,
    compute_score_cache_key,
    load_resume_markdown,
    load_user_profile,
    run_scoring_pipeline,
)

__all__ = [
    "DiscoverySummary",
    "RuleFilterDecision",
    "ScoringSummary",
    "add_company_from_target",
    "build_redacted_profile_summary",
    "canonicalize_url",
    "compute_dedupe_cluster_id",
    "compute_jd_hash",
    "compute_score_cache_key",
    "evaluate_rule_filter",
    "load_resume_markdown",
    "load_user_profile",
    "normalize_raw_job",
    "normalize_text",
    "redact_pii",
    "run_discovery_pipeline",
    "run_scoring_pipeline",
    "sync_watchlist_from_yaml",
]
