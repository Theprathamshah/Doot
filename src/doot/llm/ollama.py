"""Optional local Ollama LLMProvider stub (not required when using Gemini API)."""

from doot.models.job import JobPosting, ScoreResult, TailoredAssets


class OllamaLLMProvider:
    """Placeholder adapter for future local Ollama inference."""

    provider_name = "ollama"

    def __init__(self, *, model_name: str = "qwen2.5:7b-instruct") -> None:
        self.model_name = model_name

    async def score_job(self, redacted_profile: str, untrusted_jd: str) -> ScoreResult:
        raise NotImplementedError("Local Ollama provider is optional and not yet enabled.")

    async def draft_tailoring(
        self,
        evidence_chunks: list[str],
        job: JobPosting,
    ) -> TailoredAssets:
        raise NotImplementedError("Local Ollama provider is optional and not yet enabled.")

    async def draft_free_text_answer(
        self,
        question: str,
        evidence_chunks: list[str],
        job: JobPosting,
    ) -> str:
        raise NotImplementedError("Local Ollama provider is optional and not yet enabled.")
