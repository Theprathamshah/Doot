"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DootSettings(BaseSettings):
    """Typed configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="DOOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/doot.db")

    # LLM configuration (Gemini API primary; zero local model downloads)
    llm_provider: str = "gemini"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash"

    # Search dorking configuration (Phase 2)
    search_provider: str = "serper"
    serper_api_key: SecretStr = SecretStr("")
    brave_api_key: SecretStr = SecretStr("")
    tavily_api_key: SecretStr = SecretStr("")

    # Optional one-way notifications (Phase 4)
    notify_method: str = "macos"
    notify_webhook_url: str = ""

    # Profile & safety guardrails
    profile_version: str = "v1"
    daily_apply_cap: int = Field(default=15, ge=1, le=100)
    per_company_cap: int = Field(default=2, ge=1, le=10)
    min_digest_score: int = Field(default=4, ge=1, le=5)
    dry_run: bool = True

    @property
    def stop_file_path(self) -> Path:
        """Path to the global kill-switch sentinel file."""
        return self.data_dir / "STOP"

    @property
    def profile_path(self) -> Path:
        """Default path to user profile JSON."""
        return self.data_dir / "profile.json"

    def secret_values(self) -> list[str]:
        """Return non-empty secret strings for log redaction."""
        candidates = [
            self.gemini_api_key.get_secret_value(),
            self.serper_api_key.get_secret_value(),
            self.brave_api_key.get_secret_value(),
            self.tavily_api_key.get_secret_value(),
        ]
        return [s for s in candidates if s]
