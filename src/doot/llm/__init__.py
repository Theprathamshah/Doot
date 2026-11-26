"""LLM providers package."""

from doot.config import DootSettings
from doot.llm.base import LLMProvider
from doot.llm.gemini import DailyQuotaExceededError, GeminiLLMProvider
from doot.llm.ollama import OllamaLLMProvider


def build_llm_provider(settings: DootSettings) -> LLMProvider:
    """Construct the configured LLMProvider."""
    provider = settings.llm_provider.strip().lower()
    if provider == "gemini":
        return GeminiLLMProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model_name=settings.gemini_model,
            fallback_models=settings.gemini_fallback_models,
            tailor_models=settings.gemini_tailor_models,
            min_interval_seconds=settings.llm_min_interval_seconds,
        )
    if provider == "ollama":
        return OllamaLLMProvider()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


__all__ = [
    "DailyQuotaExceededError",
    "GeminiLLMProvider",
    "LLMProvider",
    "OllamaLLMProvider",
    "build_llm_provider",
]
