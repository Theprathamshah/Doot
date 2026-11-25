"""Stage 1 deterministic rule filter to eliminate ineligible jobs before LLM scoring."""

import re
from dataclasses import dataclass

from doot.models.job import CompanyRecord, JobRecord
from doot.models.profile import UserProfile

_GEO_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bus[\s-]*only\b", re.IGNORECASE),
    re.compile(r"\bu\.s\.[\s-]*only\b", re.IGNORECASE),
    re.compile(r"\bunited states only\b", re.IGNORECASE),
    re.compile(r"\bmust be authorized to work in the (?:u\.s\.|us|united states)\b", re.IGNORECASE),
    re.compile(r"\b(?:us|u\.s\.) work authorization(?: is)? required\b", re.IGNORECASE),
    re.compile(
        r"\bmust reside in (?:the )?(?:us|u\.s\.|united states|uk|canada|eu)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsecurity clearance\b", re.IGNORECASE),
    re.compile(r"\b(?:uk|eu|emea|canada|europe)[\s-]*only\b", re.IGNORECASE),
    re.compile(r"\bremote\s*[-–(]\s*(?:us|usa|uk|eu|emea|north america)\b", re.IGNORECASE),
)

_SENIORITY_BLOCK_TITLES: tuple[str, ...] = (
    "intern",
    "internship",
    "staff engineer",
    "staff software",
    "principal",
    "distinguished",
    "director",
    "vp ",
    "vice president",
    "head of engineering",
    "chief architect",
    "cto",
)

_UNRELATED_TITLE_KEYWORDS: tuple[str, ...] = (
    "ios ",
    "android ",
    "flutter",
    "salesforce",
    "sap ",
    "manual qa",
    "recruiter",
    "talent acquisition",
    "account executive",
    "sales manager",
    "marketing",
    "customer support",
)

_POSITIVE_TECH_KEYWORDS: tuple[str, ...] = (
    "backend",
    "back-end",
    "full stack",
    "fullstack",
    "full-stack",
    "software engineer",
    "software developer",
    "sde",
    "platform",
    "distributed",
    "microservice",
    "aws",
    "cloud",
    "java",
    "spring",
    "node",
    "typescript",
    "react",
    "next.js",
    "python",
    "ai ",
    "ai/",
    "llm",
    "rag",
    "mcp",
    "agent",
    "api",
)

_YEARS_PATTERN = re.compile(r"\b(\d{1,2})\s*\+?\s*(?:to\s*\d{1,2}\s*)?years?\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RuleFilterDecision:
    """Outcome of Stage 1 deterministic rule filtering."""

    passed: bool
    reason: str


def _extract_max_required_yoe(text: str) -> int | None:
    """Find explicit minimum years-of-experience asks in JD text."""
    matches = [int(m.group(1)) for m in _YEARS_PATTERN.finditer(text)]
    # Ignore implausible numbers like "20 years in business" by bounding <= 20
    plausible = [y for y in matches if 1 <= y <= 15]
    return max(plausible) if plausible else None


def evaluate_rule_filter(
    job: JobRecord,
    profile: UserProfile,
    *,
    company: CompanyRecord | None = None,
    blocked_companies: frozenset[str] = frozenset(),
    yoe_buffer: float = 3.0,
) -> RuleFilterDecision:
    """Apply zero-cost eligibility rules (blocklist, India/Remote geo, seniority, stack)."""
    if company is not None and company.is_blocklisted:
        return RuleFilterDecision(False, f"blocklisted company: {company.name}")

    if job.normalized_company in blocked_companies:
        return RuleFilterDecision(False, f"blocklisted company: {job.company_name}")

    if not job.india_eligible:
        return RuleFilterDecision(
            False, f"location not India/remote eligible: '{job.location_raw}'"
        )

    combined_geo_text = f"{job.location_raw}\n{job.jd_text}"
    for pattern in _GEO_BLOCK_PATTERNS:
        if pattern.search(combined_geo_text):
            return RuleFilterDecision(
                False, f"geo/visa restriction matched: '{pattern.pattern}'"
            )

    title_lower = f" {job.title.lower()} "
    for blocked_level in _SENIORITY_BLOCK_TITLES:
        if blocked_level in title_lower:
            return RuleFilterDecision(
                False, f"seniority mismatch in title: '{blocked_level.strip()}'"
            )

    for unrelated in _UNRELATED_TITLE_KEYWORDS:
        if unrelated in title_lower:
            return RuleFilterDecision(False, f"unrelated role in title: '{unrelated.strip()}'")

    max_yoe_ask = _extract_max_required_yoe(job.jd_text)
    if max_yoe_ask is not None and max_yoe_ask > (profile.years_of_experience + yoe_buffer):
        return RuleFilterDecision(
            False,
            f"experience requirement too high ({max_yoe_ask}+ yrs vs "
            f"{profile.years_of_experience} yrs)",
        )

    haystack = f" {job.title.lower()} {job.jd_text.lower()} "
    if not any(kw in haystack for kw in _POSITIVE_TECH_KEYWORDS):
        return RuleFilterDecision(False, "no overlap with target backend/fullstack/AWS/AI stack")

    return RuleFilterDecision(True, "passed stage 1 rules")
