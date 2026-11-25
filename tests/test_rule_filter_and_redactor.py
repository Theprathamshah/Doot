"""Offline tests for Stage 1 rule filter and PII redactor."""

from doot.models import ATSProvider, CompanyRecord, JobRecord, ProfileLinks, UserProfile
from doot.pipeline.redactor import build_redacted_profile_summary, redact_pii
from doot.pipeline.rule_filter import evaluate_rule_filter


def _sample_profile() -> UserProfile:
    return UserProfile(
        full_name="Pratham Shah",
        email="pratham.dev@example.com",
        phone="+91-9876543210",
        location="Ahmedabad, India",
        work_authorization_india="Authorized to work in India",
        requires_us_visa=True,
        years_of_experience=2.5,
        current_ctc_lpa=12.0,
        expected_ctc_lpa=22.0,
        notice_period_days=30,
        preferred_work_modes=["remote", "hybrid"],
        preferred_locations=["Bengaluru", "Remote", "India"],
        target_roles=["Backend Engineer", "AI Engineer"],
        links=ProfileLinks(
            linkedin="https://linkedin.com/in/prathamshah",
            github="https://github.com/prathamshah",
            portfolio="https://pratham.dev",
        ),
    )


def _sample_job_record(
    *,
    title: str = "Backend Engineer (AWS / Java)",
    location_raw: str = "Bengaluru, India (Hybrid)",
    jd_text: str = "Looking for 2+ years in Java, Spring Boot, AWS Lambda.",
    india_eligible: bool = True,
) -> JobRecord:
    return JobRecord(
        id=1,
        company_id=10,
        source="greenhouse",
        external_job_id="1001",
        ats_provider=ATSProvider.GREENHOUSE,
        board_token="acme",
        canonical_url="https://boards.greenhouse.io/acme/jobs/1001",
        apply_url="https://boards.greenhouse.io/acme/jobs/1001#app",
        title=title,
        normalized_title=title.lower(),
        company_name="Acme",
        normalized_company="acme",
        location_raw=location_raw,
        is_remote="remote" in location_raw.lower(),
        india_eligible=india_eligible,
        alert_only=False,
        jd_text=jd_text,
        jd_hash="h1",
        dedupe_cluster_id="c1",
        discovered_at="2026-11-25T00:00:00Z",
        updated_at="2026-11-25T00:00:00Z",
    )


def test_rule_filter_blocks_us_only_seniority_blocklist_and_unrelated() -> None:
    profile = _sample_profile()

    # Blocklisted company
    blocked = CompanyRecord(
        id=10,
        name="Acme",
        normalized_name="acme",
        ats_provider=ATSProvider.GREENHOUSE,
        board_token="acme",
        is_blocklisted=True,
    )
    assert evaluate_rule_filter(_sample_job_record(), profile, company=blocked).passed is False

    # Remote US-only trap inside JD body
    us_trap = _sample_job_record(
        location_raw="Remote",
        jd_text="Great AWS backend role. Must be authorized to work in the US (US only).",
    )
    res_geo = evaluate_rule_filter(us_trap, profile)
    assert res_geo.passed is False
    assert "geo/visa" in res_geo.reason

    # Staff / 8+ years traps
    staff = _sample_job_record(title="Staff Software Engineer")
    assert evaluate_rule_filter(staff, profile).passed is False

    high_yoe = _sample_job_record(jd_text="Minimum 8+ years of Java and AWS experience required.")
    res_yoe = evaluate_rule_filter(high_yoe, profile)
    assert res_yoe.passed is False
    assert "experience requirement too high" in res_yoe.reason

    # Unrelated role
    unrelated = _sample_job_record(
        title="Salesforce Administrator", jd_text="Manage CRM workflows."
    )
    assert evaluate_rule_filter(unrelated, profile).passed is False

    # Valid India / Remote match passes
    good = _sample_job_record(
        title="Software Engineer (AWS & LLM/RAG)",
        location_raw="Remote - India",
        jd_text="2-4 years experience with AWS Lambda, Python, TypeScript, RAG, MCP.",
    )
    assert evaluate_rule_filter(good, profile).passed is True


def test_pii_redactor_scrubs_all_personal_identifiers() -> None:
    profile = _sample_profile()
    raw_resume = (
        "Pratham Shah\n"
        "Email: pratham.dev@example.com | Phone: +91-9876543210 or 9876543210\n"
        "GitHub: https://github.com/prathamshah | LinkedIn: https://linkedin.com/in/prathamshah\n"
        "Built AWS Lambda microservices handling 100K+ daily transactions.\n"
    )
    summary = build_redacted_profile_summary(profile, raw_resume)
    assert "Pratham" not in summary
    assert "Shah" not in summary
    assert "pratham.dev@example.com" not in summary
    assert "9876543210" not in summary
    assert "linkedin.com/in/prathamshah" not in summary
    assert "github.com/prathamshah" not in summary
    assert "[REDACTED_NAME]" in summary
    assert "[REDACTED_EMAIL]" in summary
    assert "[REDACTED_PHONE]" in summary
    assert "[REDACTED_URL]" in summary
    assert "AWS Lambda microservices" in summary

    # Fallback regex catches non-profile email/phone too
    extra = redact_pii("Contact alt@other.io or +1-415-555-0199", profile)
    assert "alt@other.io" not in extra
    assert "415-555-0199" not in extra
