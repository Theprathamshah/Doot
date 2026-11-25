"""Domain models for jobs, companies, scoring, and tailored assets."""

from pydantic import BaseModel, Field, field_validator

from doot.models.enums import ATSProvider, CompanySource, JobStatus


class RawJob(BaseModel):
    """Unnormalized job payload emitted by any collector."""

    source: str
    external_job_id: str
    title: str
    company_name: str
    canonical_url: str
    apply_url: str
    location_raw: str = ""
    jd_text: str = ""
    salary_raw: str | None = None
    ats_provider: ATSProvider | None = None
    board_token: str | None = None
    alert_only: bool = False


class JobPosting(BaseModel):
    """Normalized job posting ready for persistence and triage."""

    company_id: int | None = None
    source: str
    external_job_id: str
    ats_provider: ATSProvider | None = None
    board_token: str | None = None
    canonical_url: str
    apply_url: str
    title: str
    normalized_title: str
    company_name: str
    normalized_company: str
    location_raw: str = ""
    is_remote: bool = False
    india_eligible: bool = True
    alert_only: bool = False
    salary_raw: str | None = None
    jd_text: str = ""
    jd_hash: str
    dedupe_cluster_id: str


class JobRecord(JobPosting):
    """Persisted job row including lifecycle and scoring fields."""

    id: int
    status: JobStatus = JobStatus.DISCOVERED
    rule_filter_reason: str | None = None
    llm_score: int | None = None
    llm_reasoning: str | None = None
    missing_skills: list[str] = Field(default_factory=list)
    should_apply: bool | None = None
    score_cache_key: str | None = None
    discovered_at: str
    updated_at: str


class CompanyRecord(BaseModel):
    """Persisted company watchlist entry."""

    id: int
    name: str
    normalized_name: str
    ats_provider: ATSProvider
    board_token: str
    careers_url: str | None = None
    source: CompanySource = CompanySource.MANUAL
    is_blocklisted: bool = False
    is_active: bool = True
    consecutive_failures: int = 0
    last_crawled_at: str | None = None


class ScoreResult(BaseModel):
    """Structured output contract returned by an LLMProvider."""

    score: int = Field(ge=1, le=5)
    should_apply: bool
    reasoning: str = Field(max_length=500)
    missing_skills: list[str] = Field(default_factory=list)
    india_remote_eligible: bool = True
    detected_seniority: str = "mid"

    @field_validator("reasoning", mode="before")
    @classmethod
    def _clamp_reasoning(cls, value: object) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= 500:
            return text
        return text[:497].rstrip() + "..."


class TailoredAssets(BaseModel):
    """RAG-grounded tailored application assets."""

    tailored_summary: str
    cover_note: str
    outreach_dm: str
    cited_story_ids: list[str] = Field(default_factory=list)
