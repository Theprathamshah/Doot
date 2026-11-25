"""Assisted apply package (field resolution, challenge detection, Playwright)."""

from doot.applier.base import Applier, FilledFormResult, FormField, SubmitResult
from doot.applier.challenge import detect_anti_bot_challenge
from doot.applier.field_resolver import (
    AliasesConfig,
    FieldResolver,
    is_llm_allowed_for_label,
    load_aliases_config,
    normalize_label,
)
from doot.applier.playwright_applier import (
    extract_form_fields_from_html,
    load_selectors_config,
    resolve_canonical_apply_url,
    run_assisted_apply,
)

__all__ = [
    "AliasesConfig",
    "Applier",
    "FieldResolver",
    "FilledFormResult",
    "FormField",
    "SubmitResult",
    "detect_anti_bot_challenge",
    "extract_form_fields_from_html",
    "is_llm_allowed_for_label",
    "load_aliases_config",
    "load_selectors_config",
    "normalize_label",
    "resolve_canonical_apply_url",
    "run_assisted_apply",
]
