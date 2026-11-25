"""User profile and form-field resolution models."""

from pathlib import Path

from pydantic import BaseModel, Field


class ProfileLinks(BaseModel):
    """Public profile links for applications."""

    linkedin: str = ""
    github: str = ""
    portfolio: str = ""


class UserProfile(BaseModel):
    """Structured applicant profile loaded from `profile.json`."""

    profile_version: str = "v1"
    full_name: str
    email: str
    phone: str
    location: str
    work_authorization_india: str
    requires_us_visa: bool = True
    years_of_experience: float = Field(ge=0.0)
    current_ctc_lpa: float = Field(ge=0.0)
    expected_ctc_lpa: float = Field(ge=0.0)
    notice_period_days: int = Field(ge=0)
    notice_negotiable: bool = True
    preferred_work_modes: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    links: ProfileLinks = Field(default_factory=ProfileLinks)
    resume_pdf_path: Path = Path("./data/resume.pdf")
    resume_md_path: Path = Path("./data/resume.md")
    stories_md_path: Path = Path("./data/stories.md")
