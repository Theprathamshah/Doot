"""Applier Protocol and form-filling data contracts."""

from typing import Protocol

from pydantic import BaseModel, Field

from doot.models.enums import ATSProvider, JobStatus
from doot.models.job import JobPosting, TailoredAssets
from doot.models.profile import UserProfile


class FormField(BaseModel):
    """Detected interactive field on an ATS application form."""

    label: str
    field_type: str = "text"  # text | textarea | select | file | checkbox | radio
    required: bool = False
    selector_hint: str | None = None


class FilledFormResult(BaseModel):
    """Result of preparing/filling an application form prior to submission."""

    resolved_fields: dict[str, str] = Field(default_factory=dict)
    unknown_questions: list[str] = Field(default_factory=list)
    screenshot_path: str | None = None
    challenge_detected: str | None = None


class SubmitResult(BaseModel):
    """Outcome of attempting or completing an assisted application."""

    submitted: bool
    status: JobStatus
    message: str


class Applier(Protocol):
    """Interface for assisted ATS form preparation and submission."""

    ats_provider: ATSProvider

    async def prepare(
        self,
        job: JobPosting,
        profile: UserProfile,
        assets: TailoredAssets | None = None,
    ) -> FilledFormResult:
        """Inspect and pre-fill an ATS form without submitting."""
        ...

    async def submit(
        self,
        job: JobPosting,
        prepared: FilledFormResult,
    ) -> SubmitResult:
        """Complete submission after explicit human approval and guardrail checks."""
        ...
