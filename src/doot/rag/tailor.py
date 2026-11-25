"""Grounded resume tailoring and recruiter outreach generation."""

from doot.config import DootSettings
from doot.db.repository import DootRepository
from doot.llm.base import LLMProvider
from doot.models.job import JobPosting, TailoredAssets
from doot.models.profile import UserProfile
from doot.pipeline.redactor import redact_pii
from doot.rag.retriever import BM25Retriever, EvidenceChunk, parse_markdown_evidence


def load_stories_markdown(profile: UserProfile, settings: DootSettings) -> str:
    """Read optional stories.md referenced by profile or data_dir."""
    candidates = [profile.stories_md_path, settings.data_dir / "stories.md"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


def collect_evidence_chunks(
    profile: UserProfile,
    *,
    resume_markdown: str = "",
    stories_markdown: str = "",
) -> list[EvidenceChunk]:
    """Parse and PII-redact verified chunks from resume.md and stories.md."""
    chunks: list[EvidenceChunk] = []
    if stories_markdown.strip():
        redacted_stories = redact_pii(stories_markdown, profile)
        chunks.extend(parse_markdown_evidence(redacted_stories, source_file="stories.md"))
    if resume_markdown.strip():
        redacted_resume = redact_pii(resume_markdown, profile)
        chunks.extend(parse_markdown_evidence(redacted_resume, source_file="resume.md"))
    return chunks


async def tailor_for_job(
    repo: DootRepository,
    llm_provider: LLMProvider,
    *,
    job_id: int,
    profile: UserProfile,
    resume_markdown: str = "",
    stories_markdown: str = "",
    top_k: int = 5,
    persist: bool = True,
) -> TailoredAssets:
    """Retrieve matching verified evidence and generate grounded TailoredAssets."""
    job = await repo.get_job(job_id)
    if job is None:
        raise LookupError(f"Job #{job_id} not found")

    chunks = collect_evidence_chunks(
        profile,
        resume_markdown=resume_markdown,
        stories_markdown=stories_markdown,
    )
    if not chunks:
        raise ValueError(
            "No resume.md or stories.md evidence found. "
            "Create data/resume.md or data/stories.md before tailoring."
        )

    retriever = BM25Retriever(chunks)
    query = f"{job.title} {job.jd_text[:3000]}"
    selected = retriever.retrieve(query, top_k=top_k)
    valid_ids = {c.chunk_id for c in selected}
    evidence_blocks = [c.render_for_prompt() for c in selected]

    posting = JobPosting.model_validate(job.model_dump())
    assets = await llm_provider.draft_tailoring(evidence_blocks, posting)

    # Filter out any unverified citation IDs so output remains strictly grounded
    verified_citations = [cid for cid in assets.cited_story_ids if cid in valid_ids]
    grounded_assets = assets.model_copy(update={"cited_story_ids": verified_citations})

    if persist:
        await repo.upsert_application_tailoring(
            job_id,
            grounded_assets,
            resume_path=str(profile.resume_pdf_path),
        )
    return grounded_assets
