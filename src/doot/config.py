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
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_models: list[str] = Field(
        default_factory=lambda: [
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3-flash-preview",
            "gemini-4-hydrogen",
            "gemini-2.5-flash",
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
        ]
    )
    gemini_tailor_models: list[str] = Field(
        default_factory=lambda: [
            "gemini-3.8-flash",
            "gemini-4-hydrogen",
            "gemini-3.6-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it",
        ]
    )
    llm_min_interval_seconds: float = Field(default=4.2, ge=0.0)
    fallback_to_local_on_quota: bool = True

    # Discovery configuration (Phase 2)
    search_provider: str = "serper"
    serper_api_key: SecretStr = SecretStr("")
    brave_api_key: SecretStr = SecretStr("")
    tavily_api_key: SecretStr = SecretStr("")
    ats_max_consecutive_failures: int = Field(default=5, ge=1, le=50)
    ats_min_interval_seconds: float = Field(default=1.0, ge=0.0)
    dork_days: int = Field(default=7, ge=1, le=30)
    dork_max_results: int = Field(default=20, ge=1, le=50)
    jobspy_enabled: bool = False
    jobspy_sites: list[str] = Field(default_factory=lambda: ["linkedin", "indeed", "naukri"])
    jobspy_search_terms: list[str] = Field(
        default_factory=lambda: [
            "Backend Software Engineer AWS",
            "Full Stack Engineer Java Node",
            "AI Engineer LLM RAG",
        ]
    )
    jobspy_location: str = "India"
    jobspy_hours_old: int = Field(default=24, ge=1, le=168)
    jobspy_results_wanted: int = Field(default=20, ge=1, le=100)
    jobspy_timeout_seconds: float = Field(default=45.0, ge=5.0)

    # Optional one-way notifications (Phase 4)
    notify_method: str = "macos"  # macos | webhook | email | none
    notify_webhook_url: str = ""
    notify_smtp_host: str = ""
    notify_smtp_port: int = 587
    notify_email_from: str = ""
    notify_email_to: str = ""
    notify_email_password: SecretStr = SecretStr("")

    # Profile & safety guardrails
    profile_version: str = "v1"
    daily_apply_cap: int = Field(default=15, ge=1, le=100)
    per_company_cap: int = Field(default=2, ge=1, le=10)
    daily_outreach_cap: int = Field(default=10, ge=1, le=50)
    min_digest_score: int = Field(default=4, ge=1, le=5)
    dry_run: bool = True
    hunter_api_key: SecretStr = SecretStr("")

    @property
    def stop_file_path(self) -> Path:
        """Path to the global kill-switch sentinel file."""
        return self.data_dir / "STOP"

    @property
    def profile_path(self) -> Path:
        """Default path to user profile JSON."""
        return self.data_dir / "profile.json"

    @property
    def companies_path(self) -> Path:
        """Default path to companies watchlist YAML."""
        return self.data_dir / "companies.yaml"

    @property
    def dork_queries_path(self) -> Path:
        """Default path to search dork queries YAML."""
        return self.data_dir / "dork_queries.yaml"

    def secret_values(self) -> list[str]:
        """Return non-empty secret strings for log redaction."""
        candidates = [
            self.gemini_api_key.get_secret_value(),
            self.serper_api_key.get_secret_value(),
            self.brave_api_key.get_secret_value(),
            self.tavily_api_key.get_secret_value(),
            self.notify_email_password.get_secret_value(),
            self.hunter_api_key.get_secret_value(),
        ]
        return [s for s in candidates if s]
