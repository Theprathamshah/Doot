"""LLM provider Protocol contract."""

from typing import Protocol

from doot.models.job import JobPosting, ScoreResult, TailoredAssets


class LLMProvider(Protocol):
    """Interface for LLM inference providers (Gemini primary, Ollama optional)."""

    provider_name: str
    model_name: str

    async def score_job(self, redacted_profile: str, untrusted_jd: str) -> ScoreResult:
        """Score a job posting against a PII-redacted candidate profile."""
        ...

    async def draft_tailoring(
        self,
        evidence_chunks: list[str],
        job: JobPosting,
    ) -> TailoredAssets:
        """Generate grounded tailored summary, cover note, and outreach DM."""
        ...

    async def draft_free_text_answer(
        self,
        question: str,
        evidence_chunks: list[str],
        job: JobPosting,
    ) -> str:
        """Draft a safe free-text answer (motivation/cover only)."""
        ...
