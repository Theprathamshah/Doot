"""Two-stage filtering and cached LLM scoring pipeline."""

import hashlib
from dataclasses import dataclass, field

from loguru import logger

from doot.config import DootSettings
from doot.db.repository import DootRepository
from doot.llm import LLMProvider, build_llm_provider
from doot.models.enums import JobStatus, RunStatus, RunType
from doot.models.profile import UserProfile
from doot.pipeline.redactor import build_redacted_profile_summary
from doot.pipeline.rule_filter import evaluate_rule_filter


@dataclass(slots=True)
class ScoringSummary:
    """Aggregated counts returned after a scoring run."""

    evaluated: int = 0
    filtered_out: int = 0
    scored: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def load_user_profile(settings: DootSettings) -> UserProfile:
    """Load and validate UserProfile from data/profile.json."""
    if not settings.profile_path.exists():
        raise FileNotFoundError(
            f"Profile not found at {settings.profile_path}. "
            "Copy data/profile.example.json to data/profile.json first."
        )
    return UserProfile.model_validate_json(settings.profile_path.read_text(encoding="utf-8"))


def load_resume_markdown(profile: UserProfile, settings: DootSettings) -> str:
    """Read optional resume.md referenced by profile or data_dir."""
    candidates = [profile.resume_md_path, settings.data_dir / "resume.md"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


def compute_score_cache_key(jd_hash: str, profile_version: str, model_name: str) -> str:
    """Build deterministic cache key for LLM scoring results."""
    raw = f"{jd_hash}:{profile_version}:{model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def run_scoring_pipeline(
    settings: DootSettings,
    repo: DootRepository,
    *,
    profile: UserProfile,
    resume_markdown: str = "",
    llm_provider: LLMProvider | None = None,
    limit: int = 200,
    stage1_only: bool = False,
) -> ScoringSummary:
    """Evaluate DISCOVERED jobs through Stage 1 rules and Stage 2 cached LLM scoring."""
    run_id = await repo.start_run(RunType.SCORE)
    summary = ScoringSummary()

    try:
        discovered = await repo.list_jobs(status=JobStatus.DISCOVERED, limit=limit)
        companies_by_id = {c.id: c for c in await repo.list_companies(active_only=False)}
        blocked_names = frozenset(
            c.normalized_name for c in companies_by_id.values() if c.is_blocklisted
        )

        active_llm: LLMProvider | None = None
        if not stage1_only and discovered:
            active_llm = llm_provider or build_llm_provider(settings)

        redacted_profile = build_redacted_profile_summary(profile, resume_markdown)
        model_name = active_llm.model_name if active_llm is not None else settings.gemini_model

        for job in discovered:
            summary.evaluated += 1
            company = companies_by_id.get(job.company_id) if job.company_id is not None else None
            decision = evaluate_rule_filter(
                job,
                profile,
                company=company,
                blocked_companies=blocked_names,
            )
            if not decision.passed:
                await repo.mark_job_filtered_out(job.id, decision.reason)
                summary.filtered_out += 1
                continue

            if stage1_only or active_llm is None:
                continue

            cache_key = compute_score_cache_key(job.jd_hash, profile.profile_version, model_name)
            cached = await repo.get_cached_score(cache_key)
            if cached is not None:
                await repo.save_job_score(job.id, cached, score_cache_key=cache_key, actor="cache")
                summary.cache_hits += 1
                summary.scored += 1
                continue

            untrusted_payload = (
                f"Role Title: {job.title}\n"
                f"Company: {job.company_name}\n"
                f"Location: {job.location_raw}\n\n"
                f"{job.jd_text}"
            )
            try:
                result = await active_llm.score_job(redacted_profile, untrusted_payload)
            except Exception as exc:
                summary.failed += 1
                summary.errors.append(f"Job #{job.id} ({job.title}): {exc}")
                logger.warning("Scoring failed for job #{}: {}", job.id, exc)
                await repo.transition_job(
                    job.id,
                    JobStatus.FAILED,
                    actor="scorer",
                    reason=str(exc)[:300],
                )
                continue

            await repo.save_job_score(job.id, result, score_cache_key=cache_key, actor="scorer")
            summary.llm_calls += 1
            summary.scored += 1

        run_status = RunStatus.PARTIAL if summary.failed > 0 else RunStatus.SUCCESS
        await repo.finish_run(
            run_id,
            status=run_status,
            stats={
                "evaluated": summary.evaluated,
                "filtered_out": summary.filtered_out,
                "scored": summary.scored,
                "cache_hits": summary.cache_hits,
                "llm_calls": summary.llm_calls,
                "failed": summary.failed,
            },
        )
        return summary
    except Exception as exc:
        await repo.finish_run(run_id, status=RunStatus.FAILED, error_message=str(exc))
        raise
