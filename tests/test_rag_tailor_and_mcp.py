"""Offline tests for BM25 evidence retrieval, grounded tailoring, and MCP server."""

import json
from types import SimpleNamespace

from doot.config import DootSettings
from doot.db import DootRepository, initialize_schema, open_connection
from doot.llm.gemini import GeminiLLMProvider
from doot.mcp_server import (
    create_mcp_server,
    mcp_get_funnel_stats,
    mcp_get_job_details,
    mcp_list_top_jobs,
    mcp_mark_job_status,
    mcp_tailor_for_job,
)
from doot.models import (
    JobPosting,
    JobStatus,
    ProfileLinks,
    ScoreResult,
    TailoredAssets,
    UserProfile,
)
from doot.rag import BM25Retriever, parse_markdown_evidence, tailor_for_job
from tests.conftest import make_sample_job


def _sample_profile(tmp_settings: DootSettings) -> UserProfile:
    return UserProfile(
        full_name="Pratham Shah",
        email="pratham@example.com",
        phone="+91-9876543210",
        location="Ahmedabad, India",
        work_authorization_india="Authorized in India",
        years_of_experience=2.5,
        current_ctc_lpa=12.0,
        expected_ctc_lpa=22.0,
        notice_period_days=30,
        links=ProfileLinks(github="https://github.com/pratham"),
        resume_pdf_path=tmp_settings.data_dir / "resume.pdf",
        resume_md_path=tmp_settings.data_dir / "resume.md",
        stories_md_path=tmp_settings.data_dir / "stories.md",
    )


_STORIES_MD = """
### [story:lambda-30x] AWS Lambda Performance Optimization
Reduced AWS Lambda receipt-processing latency 30x (30s to 1s) via MySQL batching and EventBridge.

### [story:voxi-wasm] Client-Heavy Audio SaaS
Compiled FFmpeg and RNNoise into WebAssembly (WASM) in Next.js, removing 100% of GPU server costs.
"""


def test_bm25_retriever_ranks_matching_story_first() -> None:
    chunks = parse_markdown_evidence(_STORIES_MD, source_file="stories.md")
    assert [c.chunk_id for c in chunks] == ["lambda-30x", "voxi-wasm"]

    retriever = BM25Retriever(chunks)
    aws_hits = retriever.retrieve("Senior AWS Lambda and EventBridge backend role", top_k=1)
    assert aws_hits[0].chunk_id == "lambda-30x"

    wasm_hits = retriever.retrieve("Frontend Next.js WebAssembly FFmpeg audio engineer", top_k=1)
    assert wasm_hits[0].chunk_id == "voxi-wasm"


async def test_tailor_for_job_filters_unverified_citations_and_persists(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    profile = _sample_profile(tmp_settings)
    job = make_sample_job(external_job_id="tailor-1")
    job.jd_text = "Seeking AWS Lambda and EventBridge microservice developer."
    job_id, _ = await repo.upsert_job(job)

    class _FakeTailorLLM:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"

        def __init__(self) -> None:
            self.seen_chunks: list[str] = []

        async def score_job(self, redacted_profile: str, untrusted_jd: str) -> ScoreResult:
            raise NotImplementedError

        async def draft_tailoring(
            self, evidence_chunks: list[str], job_posting: JobPosting
        ) -> TailoredAssets:
            self.seen_chunks = evidence_chunks
            return TailoredAssets(
                tailored_summary="Backend engineer who optimized AWS Lambda 30x.",
                cover_note="My experience cutting Lambda latency from 30s to 1s fits this role.",
                outreach_dm="Hi! Built 30x faster AWS Lambda pipelines and 100K/day services.",
                cited_story_ids=["lambda-30x", "invented-hallucinated-story"],
            )

        async def draft_free_text_answer(
            self, question: str, evidence_chunks: list[str], job_posting: JobPosting
        ) -> str:
            raise NotImplementedError

    fake_llm = _FakeTailorLLM()
    assets = await tailor_for_job(
        repo,
        fake_llm,
        job_id=job_id,
        profile=profile,
        resume_markdown="Pratham Shah | pratham@example.com\n## Skills\nJava, AWS Lambda",
        stories_markdown=_STORIES_MD,
        top_k=2,
    )

    # Hallucinated citation ID must be stripped; PII must not reach LLM chunks
    assert assets.cited_story_ids == ["lambda-30x"]
    joined_chunks = "\n".join(fake_llm.seen_chunks)
    assert "Pratham" not in joined_chunks
    assert "pratham@example.com" not in joined_chunks

    stored_app = await repo.get_application(job_id)
    assert stored_app is not None
    assert "30x" in stored_app["tailored_summary"]


async def test_gemini_draft_tailoring_structured_output() -> None:
    captured: dict[str, object] = {}

    class _FakeModels:
        async def generate_content(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                parsed=None,
                text=json.dumps(
                    {
                        "tailored_summary": "AWS + Java backend engineer.",
                        "cover_note": "Delivered 30x Lambda speedup.",
                        "outreach_dm": "Hi, saw your AWS role.",
                        "cited_story_ids": ["lambda-30x"],
                    }
                ),
            )

    fake_client = SimpleNamespace(aio=SimpleNamespace(models=_FakeModels()))
    provider = GeminiLLMProvider(api_key="dummy", model_name="gemini-2.5-flash", client=fake_client)
    job = make_sample_job(external_job_id="g-tailor")
    assets = await provider.draft_tailoring(["[lambda-30x] 30x speedup"], job)
    assert assets.cited_story_ids == ["lambda-30x"]
    assert "<untrusted_jd>" in str(captured["contents"])


async def test_mcp_server_tools_and_state_updates(tmp_settings: DootSettings) -> None:
    profile = _sample_profile(tmp_settings)
    (tmp_settings.data_dir / "profile.json").write_text(
        profile.model_dump_json(), encoding="utf-8"
    )
    (tmp_settings.data_dir / "stories.md").write_text(_STORIES_MD, encoding="utf-8")

    async with open_connection(tmp_settings.db_path) as conn:
        await initialize_schema(conn)
        repo = DootRepository(conn)
        job_id, _ = await repo.upsert_job(make_sample_job(external_job_id="mcp-1"))
        await repo.save_job_score(
            job_id,
            ScoreResult(score=5, should_apply=True, reasoning="Exact AWS fit", missing_skills=[]),
            score_cache_key="mcp-k1",
        )

    top = await mcp_list_top_jobs(tmp_settings, min_score=4)
    assert len(top) == 1
    assert top[0]["id"] == job_id

    class _StubTailorLLM:
        provider_name = "gemini"
        model_name = "gemini-2.5-flash"

        async def score_job(self, redacted_profile: str, untrusted_jd: str) -> ScoreResult:
            raise NotImplementedError

        async def draft_tailoring(
            self, evidence_chunks: list[str], job: JobPosting
        ) -> TailoredAssets:
            return TailoredAssets(
                tailored_summary="Grounded summary",
                cover_note="Grounded cover",
                outreach_dm="Grounded DM",
                cited_story_ids=["lambda-30x"],
            )

        async def draft_free_text_answer(
            self, question: str, evidence_chunks: list[str], job: JobPosting
        ) -> str:
            raise NotImplementedError

    tailored = await mcp_tailor_for_job(tmp_settings, job_id, llm_provider=_StubTailorLLM())
    assert tailored["cited_story_ids"] == ["lambda-30x"]

    details = await mcp_get_job_details(tmp_settings, job_id)
    assert details["application"]["tailored_summary"] == "Grounded summary"

    marked = await mcp_mark_job_status(tmp_settings, job_id, "APPLIED")
    assert marked == {"id": job_id, "status": JobStatus.APPLIED.value}

    stats = await mcp_get_funnel_stats(tmp_settings)
    assert stats[JobStatus.APPLIED.value] == 1

    server = create_mcp_server(tmp_settings)
    tools = await server.list_tools()
    tool_names = {t.name for t in tools}
    assert {
        "list_top_jobs",
        "get_job_details",
        "tailor_job",
        "mark_job_status",
        "get_funnel_stats",
    }.issubset(tool_names)
