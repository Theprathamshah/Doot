"""Deterministic field resolution with strict anti-hallucination guardrails."""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from doot.db.repository import DootRepository
from doot.llm.base import LLMProvider
from doot.models.job import JobPosting, TailoredAssets
from doot.models.profile import UserProfile

AskUserFn = Callable[[str], Awaitable[str | None]]

_DEFAULT_BLOCKED_KEYWORDS: tuple[str, ...] = (
    "salary",
    "ctc",
    "compensation",
    "pay",
    "notice",
    "visa",
    "sponsorship",
    "authorized to work",
    "citizenship",
    "gender",
    "race",
    "ethnicity",
    "veteran",
    "disability",
    "eeo",
    "criminal",
    "conviction",
    "age",
)

_MOTIVATION_HINTS: tuple[str, ...] = (
    "why do you want",
    "why this company",
    "why are you interested",
    "why us",
    "why join",
    "cover letter",
    "additional information",
    "anything else you",
    "tell us about yourself",
)


@dataclass(frozen=True, slots=True)
class AliasesConfig:
    """Loaded canonical field aliases and LLM safety blocklist."""

    field_aliases: dict[str, list[str]]
    llm_blocked_keywords: tuple[str, ...]


def load_aliases_config(path: Path) -> AliasesConfig:
    """Load aliases.yaml or return safe built-in defaults."""
    if not path.exists():
        return AliasesConfig(field_aliases={}, llm_blocked_keywords=_DEFAULT_BLOCKED_KEYWORDS)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_aliases = data.get("field_aliases") or {}
    aliases = {
        str(k): [str(v).lower() for v in vals]
        for k, vals in raw_aliases.items()
        if isinstance(vals, list)
    }
    blocked = tuple(
        str(w).lower() for w in (data.get("llm_blocked_keywords") or _DEFAULT_BLOCKED_KEYWORDS)
    )
    return AliasesConfig(field_aliases=aliases, llm_blocked_keywords=blocked)


def normalize_label(label: str) -> str:
    """Normalize a form label for deterministic matching and qa_cache lookup."""
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", label.lower())
    return " ".join(cleaned.split())


def is_llm_allowed_for_label(
    label: str,
    *,
    blocked_keywords: tuple[str, ...] = _DEFAULT_BLOCKED_KEYWORDS,
) -> bool:
    """Allow LLM free-text drafting ONLY for motivation prompts; block legal/CTC/notice/visa."""
    norm = normalize_label(label)
    if any(bad in norm for bad in blocked_keywords):
        return False
    return any(hint in norm for hint in _MOTIVATION_HINTS)


def profile_canonical_map(profile: UserProfile) -> dict[str, str]:
    """Flatten UserProfile into canonical string values for exact/alias field matching."""
    name_parts = profile.full_name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    return {
        "full_name": profile.full_name,
        "first_name": first_name,
        "last_name": last_name,
        "email": profile.email,
        "phone": profile.phone,
        "location": profile.location,
        "linkedin": profile.links.linkedin,
        "github": profile.links.github,
        "portfolio": profile.links.portfolio,
        "current_ctc_lpa": f"{profile.current_ctc_lpa:g} LPA",
        "expected_ctc_lpa": f"{profile.expected_ctc_lpa:g} LPA",
        "notice_period_days": str(profile.notice_period_days),
        "years_of_experience": f"{profile.years_of_experience:g}",
        "work_authorization_india": profile.work_authorization_india,
    }


class FieldResolver:
    """Resolves ATS form labels in strict order: profile -> aliases -> qa_cache -> LLM/CLI."""

    def __init__(
        self,
        *,
        profile: UserProfile,
        aliases: AliasesConfig,
        repo: DootRepository,
        llm_provider: LLMProvider | None = None,
        evidence_chunks: list[str] | None = None,
        ask_user_fn: AskUserFn | None = None,
    ) -> None:
        self.profile = profile
        self.aliases = aliases
        self.repo = repo
        self.llm_provider = llm_provider
        self.evidence_chunks = evidence_chunks or []
        self.ask_user_fn = ask_user_fn
        self._profile_values = profile_canonical_map(profile)

    def _match_profile_or_alias(self, label: str) -> str | None:
        norm = normalize_label(label)
        if not norm:
            return None

        # 1. Exact canonical key match (e.g. "email", "phone", "full name")
        for key, value in self._profile_values.items():
            if norm == key.replace("_", " ") and value:
                return value

        # 2. Alias substring match from aliases.yaml
        for canonical_key, phrases in self.aliases.field_aliases.items():
            if any(phrase in norm for phrase in phrases):
                value = self._profile_values.get(canonical_key, "")
                if value:
                    return value
        return None

    async def resolve_field(
        self,
        label: str,
        *,
        job: JobPosting,
        tailored_assets: TailoredAssets | None = None,
        cache_unknown_answers: bool = True,
    ) -> str | None:
        """Resolve a single field value according to the anti-hallucination hierarchy."""
        direct = self._match_profile_or_alias(label)
        if direct is not None:
            return direct

        norm = normalize_label(label)
        if not norm:
            return None

        # 3. Confirmed qa_cache lookup (company-specific first, then global)
        cached = await self.repo.lookup_qa_cache(norm, company_id=job.company_id)
        if cached is not None:
            return cached

        # 4. Motivation / cover-note prompt -> use tailored cover_note or safe LLM draft
        if is_llm_allowed_for_label(label, blocked_keywords=self.aliases.llm_blocked_keywords):
            if "cover letter" in norm and tailored_assets and tailored_assets.cover_note:
                return tailored_assets.cover_note
            if self.llm_provider is not None and self.evidence_chunks:
                return await self.llm_provider.draft_free_text_answer(
                    label, self.evidence_chunks, job
                )

        # 5. Escalate unknown question to user in CLI and store in qa_cache
        if self.ask_user_fn is not None:
            answer = await self.ask_user_fn(label)
            if answer is not None and answer.strip():
                cleaned = answer.strip()
                if cache_unknown_answers:
                    await self.repo.upsert_qa_cache(
                        question_normalized=norm,
                        question_sample=label.strip(),
                        answer=cleaned,
                        scope="global",
                        confirmed_by_user=True,
                    )
                return cleaned

        return None
