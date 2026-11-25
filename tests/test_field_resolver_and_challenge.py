"""Offline tests for FieldResolver anti-hallucination hierarchy and challenge detection."""

from pathlib import Path

from doot.applier import (
    FieldResolver,
    detect_anti_bot_challenge,
    is_llm_allowed_for_label,
    load_aliases_config,
)
from doot.db import DootRepository
from doot.models import JobPosting, ProfileLinks, ScoreResult, TailoredAssets, UserProfile
from tests.conftest import make_sample_job


def _profile() -> UserProfile:
    return UserProfile(
        full_name="Pratham Shah",
        email="pratham@example.com",
        phone="+91-9876543210",
        location="Ahmedabad, India",
        work_authorization_india="Authorized to work in India",
        years_of_experience=2.5,
        current_ctc_lpa=12.0,
        expected_ctc_lpa=22.0,
        notice_period_days=30,
        links=ProfileLinks(
            linkedin="https://linkedin.com/in/pratham",
            github="https://github.com/pratham",
        ),
    )


def test_llm_blocked_for_compensation_notice_visa_and_eeo() -> None:
    assert is_llm_allowed_for_label("Why do you want to work at Acme?") is True
    assert is_llm_allowed_for_label("Cover Letter") is True

    assert is_llm_allowed_for_label("Expected CTC (LPA)") is False
    assert is_llm_allowed_for_label("Expected salary?") is False
    assert is_llm_allowed_for_label("Notice period in days") is False
    assert is_llm_allowed_for_label("Do you require visa sponsorship?") is False
    assert is_llm_allowed_for_label("Gender / Ethnicity (EEO)") is False


async def test_field_resolver_hierarchy_and_qa_cache(repo: DootRepository) -> None:
    aliases_path = Path(__file__).resolve().parent.parent / "data" / "aliases.yaml"
    aliases = load_aliases_config(aliases_path)
    profile = _profile()
    posting = make_sample_job(external_job_id="fr-1")

    llm_calls: list[str] = []

    class _StubLLM:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"

        async def score_job(self, redacted_profile: str, untrusted_jd: str) -> ScoreResult:
            raise NotImplementedError

        async def draft_tailoring(
            self, evidence_chunks: list[str], job: JobPosting
        ) -> TailoredAssets:
            raise NotImplementedError

        async def draft_free_text_answer(
            self, question: str, evidence_chunks: list[str], job: JobPosting
        ) -> str:
            llm_calls.append(question)
            return "Excited by Acme's AWS distributed systems scale."

    asked: list[str] = []

    async def _ask_user(question: str) -> str | None:
        asked.append(question)
        return "Yes, comfortable with hybrid in Bengaluru."

    resolver = FieldResolver(
        profile=profile,
        aliases=aliases,
        repo=repo,
        llm_provider=_StubLLM(),
        evidence_chunks=["[lambda-30x] 30x Lambda speedup"],
        ask_user_fn=_ask_user,
    )

    # 1) Profile exact + aliases
    assert await resolver.resolve_field("Email", job=posting) == "pratham@example.com"
    assert await resolver.resolve_field("Expected CTC (LPA)", job=posting) == "22 LPA"
    assert await resolver.resolve_field("Notice period (days)", job=posting) == "30"
    assert await resolver.resolve_field("First Name", job=posting) == "Pratham"

    # 2) Allowed motivation prompt -> LLM draft
    why = await resolver.resolve_field("Why do you want to join Acme?", job=posting)
    assert why == "Excited by Acme's AWS distributed systems scale."
    assert len(llm_calls) == 1

    # 3) Unknown question -> CLI escalation -> cached in qa_cache on second call
    q = "Are you open to hybrid work in Bengaluru?"
    expected_ans = "Yes, comfortable with hybrid in Bengaluru."
    assert await resolver.resolve_field(q, job=posting) == expected_ans
    assert await resolver.resolve_field(q, job=posting) == expected_ans
    assert asked == [q]


def test_detect_anti_bot_challenges() -> None:
    assert detect_anti_bot_challenge('<div class="g-recaptcha"></div>') == "reCAPTCHA"
    assert detect_anti_bot_challenge('<div class="h-captcha"></div>') == "hCaptcha"
    assert (
        detect_anti_bot_challenge("", frame_urls=["https://challenges.cloudflare.com/turnstile/v0"])
        == "Cloudflare Turnstile / Challenge"
    )
    assert (
        detect_anti_bot_challenge("Enter the Greenhouse security code sent to your email")
        == "Email Verification / Security Code"
    )
    assert detect_anti_bot_challenge("<form><input name='email' /></form>") is None
