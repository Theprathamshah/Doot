"""LLM providers package."""

from doot.config import DootSettings
from doot.llm.base import LLMProvider
from doot.llm.gemini import GeminiLLMProvider
from doot.llm.ollama import OllamaLLMProvider


def build_llm_provider(settings: DootSettings) -> LLMProvider:
    """Construct the configured LLMProvider."""
    provider = settings.llm_provider.strip().lower()
    if provider == "gemini":
        return GeminiLLMProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model_name=settings.gemini_model,
        )
    if provider == "ollama":
        return OllamaLLMProvider()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


__all__ = [
    "GeminiLLMProvider",
    "LLMProvider",
    "OllamaLLMProvider",
    "build_llm_provider",
]
