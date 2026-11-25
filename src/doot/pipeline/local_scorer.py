"""Zero-cost deterministic local heuristic scorer (instant fallback when LLM quota is hit)."""

import re

from doot.models.job import JobPosting, ScoreResult, TailoredAssets

_TECH_SKILLS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AWS", re.compile(r"\baws\b", re.I)),
    ("AWS Lambda", re.compile(r"\blambda\b", re.I)),
    ("Java", re.compile(r"\bjava\b", re.I)),
    ("Spring Boot", re.compile(r"\bspring\s*boot\b", re.I)),
    ("Node.js", re.compile(r"\bnode\.?js\b", re.I)),
    ("TypeScript", re.compile(r"\btypescript\b", re.I)),
    ("React", re.compile(r"\breact\b", re.I)),
    ("Next.js", re.compile(r"\bnext\.?js\b", re.I)),
    ("Python", re.compile(r"\bpython\b", re.I)),
    ("Microservices", re.compile(r"\bmicroservices?\b", re.I)),
    ("Distributed Systems", re.compile(r"\bdistributed\s+systems\b", re.I)),
    ("Event-Driven", re.compile(r"\bevent[\s-]driven\b", re.I)),
    ("SQL / Postgres / MySQL", re.compile(r"\b(?:sql|postgres|postgresql|mysql)\b", re.I)),
    ("Redis", re.compile(r"\bredis\b", re.I)),
    ("Docker", re.compile(r"\bdocker\b", re.I)),
    ("LLM / RAG / MCP", re.compile(r"\b(?:llm|llms|rag|mcp|ai\s+agents?)\b", re.I)),
)

_GAP_SKILLS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Rust", re.compile(r"\brust\b", re.I)),
    ("Go", re.compile(r"\b(?:golang|go\s+developer|in\s+go)\b", re.I)),
    ("Kubernetes", re.compile(r"\bkubernetes\b|\bk8s\b", re.I)),
    ("Terraform", re.compile(r"\bterraform\b", re.I)),
    ("Kafka", re.compile(r"\bkafka\b", re.I)),
    ("GraphQL", re.compile(r"\bgraphql\b", re.I)),
    ("Ruby on Rails", re.compile(r"\bruby\s+on\s+rails\b", re.I)),
    ("PHP", re.compile(r"\bphp\b", re.I)),
    (".NET / C#", re.compile(r"\b(?:\.net|c#)\b", re.I)),
)

_CORE_ROLE_RE = re.compile(
    r"\b(?:backend|back-end|full[\s-]?stack|software\s+engineer|software\s+developer|"
    r"sde|distributed\s+systems|microservices|ai\s+engineer|platform\s+engineer)\b",
    re.I,
)
_LOW_FIT_ROLE_RE = re.compile(
    r"\b(?:rust\s+engineering|embedded|ubuntu\s+systems|sustaining|field\s+engineer|"
    r"data\s+engineer|alliances)\b",
    re.I,
)


class LocalHeuristicLLMProvider:
    """Instant, zero-API-cost scorer implementing LLMProvider for offline/quota-exhausted runs."""

    provider_name = "local_heuristic"
    model_name = "heuristic-v1"

    async def score_job(self, redacted_profile: str, untrusted_jd: str) -> ScoreResult:
        """Score a job deterministically using skill overlap and title alignment."""
        profile_lower = redacted_profile.lower()
        first_line = untrusted_jd.splitlines()[0] if untrusted_jd else ""
        title_line = first_line.replace("Role Title:", "").strip()

        matched: list[str] = []
        for label, pattern in _TECH_SKILLS:
            if pattern.search(untrusted_jd):
                matched.append(label)

        missing: list[str] = []
        for label, pattern in _GAP_SKILLS:
            if pattern.search(untrusted_jd) and label.lower() not in profile_lower:
                missing.append(label)

        role_match = bool(_CORE_ROLE_RE.search(title_line or untrusted_jd[:200]))
        low_fit_title = bool(_LOW_FIT_ROLE_RE.search(title_line))

        points = len(matched) * 1.0 + (1.5 if role_match else 0.0) - (2.0 if low_fit_title else 0.0)
        points -= 0.75 * len(missing)

        if points >= 5.0 and role_match and not low_fit_title:
            score = 5
        elif points >= 2.5 and role_match and not low_fit_title:
            score = 4
        elif points >= 1.0:
            score = 3
        elif points >= 0.0:
            score = 2
        else:
            score = 1

        matched_str = ", ".join(matched[:6]) if matched else "general software engineering"
        reasoning = (
            f"Local match on {matched_str} ({len(matched)} skill overlap(s))"
            + ("; title outside primary SWE/AWS focus." if low_fit_title else ".")
        )
        return ScoreResult(
            score=score,
            should_apply=score >= 4,
            reasoning=reasoning,
            missing_skills=missing,
            india_remote_eligible=True,
            detected_seniority="mid",
        )

    async def draft_tailoring(
        self, evidence_chunks: list[str], job: JobPosting
    ) -> TailoredAssets:
        raise NotImplementedError("Local heuristic provider does not support RAG tailoring.")

    async def draft_free_text_answer(
        self, question: str, evidence_chunks: list[str], job: JobPosting
    ) -> str:
        raise NotImplementedError("Local heuristic provider does not support free-text drafting.")
