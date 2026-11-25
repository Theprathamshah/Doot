"""Offline tests for SearchProviders, Dork-to-Crawl, auto-pause, and CLI discovery."""

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from doot.cli import app
from doot.collectors.dork import (
    DorkDiscoverer,
    DuckDuckGoSearchProvider,
    FallbackSearchProvider,
    SerperSearchProvider,
    load_dork_pack,
)
from doot.collectors.jobspy_collector import JobSpyCollector
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.config import DootSettings
from doot.db import DootRepository
from doot.models import ATSProvider, CompanySource
from doot.pipeline import add_company_from_target, run_discovery_pipeline, sync_watchlist_from_yaml

FIXTURES = Path(__file__).parent / "fixtures"


@respx.mock
async def test_dork_to_crawl_registers_ats_board_and_fetches_live_jobs(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "link": "https://jobs.ashbyhq.com/gamma-ai/ashby-xyz-1",
                        "title": "AI Engineer - Gamma AI",
                        "snippet": "RAG and MCP in India Remote",
                    },
                    {
                        "link": "https://wellfound.com/jobs/98765-backend-aws",
                        "title": "Backend Engineer - StartupX",
                        "snippet": "AWS Lambda Bengaluru 25 LPA",
                    },
                ]
            },
        )
    )
    ashby_payload = json.loads((FIXTURES / "ashby_jobs.json").read_text(encoding="utf-8"))
    respx.get("https://api.ashbyhq.com/posting-api/job-board/gamma-ai").mock(
        return_value=httpx.Response(200, json=ashby_payload)
    )

    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    dork_file = tmp_settings.data_dir / "dork_queries.yaml"
    dork_file.write_text('queries:\n  - "site:jobs.ashbyhq.com India"\n', encoding="utf-8")

    async with httpx.AsyncClient() as client:
        provider = SerperSearchProvider("serper-test-key", client, limiter)
        summary = await run_discovery_pipeline(
            tmp_settings,
            repo,
            client=client,
            rate_limiter=limiter,
            use_ats=True,
            use_dork=True,
            search_provider=provider,
        )

    assert summary.new_boards_via_dork == 1
    assert summary.companies_crawled == 1
    # 1 live job from Ashby + 1 alert-only job from Wellfound
    assert summary.jobs_created == 2

    jobs = await repo.list_jobs()
    by_source = {j.source: j for j in jobs}
    assert by_source["ashby"].alert_only is False
    assert by_source["dork"].alert_only is True


@respx.mock
async def test_cross_source_dedup_and_board_auto_pause(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    gh_payload = json.loads((FIXTURES / "greenhouse_jobs.json").read_text(encoding="utf-8"))
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true").mock(
        return_value=httpx.Response(200, json=gh_payload)
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/deadboard/jobs?content=true").mock(
        return_value=httpx.Response(404)
    )

    await repo.upsert_company(
        name="Acme",
        normalized_name="acme",
        ats_provider=ATSProvider.GREENHOUSE,
        board_token="acme",
        source=CompanySource.MANUAL,
    )
    await repo.upsert_company(
        name="DeadBoard",
        normalized_name="deadboard",
        ats_provider=ATSProvider.GREENHOUSE,
        board_token="deadboard",
        source=CompanySource.MANUAL,
    )

    # JobSpy returns the exact same Greenhouse job via job_url_direct -> must dedupe
    fake_jobspy = JobSpyCollector(
        sites=["linkedin"],
        search_terms=["AWS"],
        scrape_fn=lambda **_: [
            {
                "site": "linkedin",
                "id": "li-888",
                "title": "Senior Backend Engineer (AWS)",
                "company": "Acme",
                "location": "Bengaluru, India",
                "job_url": "https://www.linkedin.com/jobs/view/888",
                "job_url_direct": "https://boards.greenhouse.io/acme/jobs/1001",
                "description": "AWS Lambda",
            }
        ],
    )

    tight_settings = tmp_settings.model_copy(update={"ats_max_consecutive_failures": 1})
    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    async with httpx.AsyncClient() as client:
        summary = await run_discovery_pipeline(
            tight_settings,
            repo,
            client=client,
            rate_limiter=limiter,
            use_ats=True,
            use_jobspy=True,
            jobspy_collector=fake_jobspy,
        )

    assert summary.companies_crawled == 1
    assert summary.companies_failed == 1
    assert summary.companies_paused == 1
    assert summary.jobs_created == 2
    assert summary.jobs_deduped_cross_source == 1

    dead = await repo.get_company_by_ats(ATSProvider.GREENHOUSE, "deadboard")
    assert dead is not None
    assert dead.is_active is False


@respx.mock
async def test_add_company_detects_embedded_html_board(repo: DootRepository) -> None:
    html_text = (FIXTURES / "careers_embed.html").read_text(encoding="utf-8")
    respx.get("https://delta.example.com/careers").mock(
        return_value=httpx.Response(200, text=html_text)
    )
    async with httpx.AsyncClient() as client:
        cid, provider, token = await add_company_from_target(
            repo,
            "https://delta.example.com/careers",
            client=client,
            name="Delta Corp",
        )
    assert cid > 0
    assert provider == ATSProvider.GREENHOUSE
    assert token == "deltaembedded"


async def test_dork_discoverer_preserves_existing_company_name(repo: DootRepository) -> None:
    await repo.upsert_company(
        name="Gamma AI Official",
        normalized_name="gamma ai official",
        ats_provider=ATSProvider.ASHBY,
        board_token="gamma-ai",
    )

    class _FakeProvider:
        provider_name = "fake"

        async def search(self, query: str, *, days: int = 7, max_results: int = 20):
            from doot.collectors.base import SearchHit

            return [
                SearchHit(
                    url="https://jobs.ashbyhq.com/gamma-ai/123",
                    title="Role",
                    provider="fake",
                )
            ]

    res = await DorkDiscoverer(_FakeProvider(), repo).discover(["q"])
    assert res.new_companies_added == []
    comp = await repo.get_company_by_ats(ATSProvider.ASHBY, "gamma-ai")
    assert comp is not None
    assert comp.name == "Gamma AI Official"


async def test_sync_watchlist_from_yaml(tmp_path: Path, repo: DootRepository) -> None:
    watchlist = tmp_path / "companies.yaml"
    watchlist.write_text(
        """
companies:
  - name: Postman
    ats_provider: greenhouse
    board_token: postman
""",
        encoding="utf-8",
    )
    synced = await sync_watchlist_from_yaml(repo, watchlist)
    assert synced == 1


def test_cli_add_company(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "cli_data"
    monkeypatch.setenv("DOOT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DOOT_DB_PATH", str(data_dir / "doot.db"))
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["add-company", "https://jobs.lever.co/cred", "--name", "CRED"],
    )
    assert res.exit_code == 0
    assert "lever:cred" in res.output


async def test_structured_dork_pack_expands_chunks_filters_titles_and_respects_priority(
    tmp_path: Path,
    repo: DootRepository,
) -> None:
    pack_path = tmp_path / "dorks.yaml"
    pack_path.write_text(
        """
global:
  freshness_days: 7
  max_results_per_query: 10
  suffix: '-intern -fresher'
  reject_title_regex: '(?i)\\b(intern|qa|manager)\\b'
  salary_signal_regex: '(?i)(₹|LPA|USD)'

ats_jobs:
  priority: 1
  queries:
    - 'site:jobs.ashbyhq.com India Backend'

social_jobs:
  priority: 3
  queries:
    - 'site:linkedin.com/posts hiring India'

target_companies:
  priority: 2
  company_chunks:
    - [Razorpay, CRED]
  templates:
    - '{companies} Backend Engineer India'
""",
        encoding="utf-8",
    )
    pack = load_dork_pack(pack_path)
    assert len(pack.queries) == 3
    assert any('("Razorpay" OR "CRED") Backend Engineer India' in q.query for q in pack.queries)

    seen_queries: list[str] = []

    class _FakeSerper:
        provider_name = "serper"

        async def search(self, query: str, *, days: int = 7, max_results: int = 20):
            from doot.collectors.base import SearchHit

            seen_queries.append(query)
            return [
                SearchHit(
                    url="https://wellfound.com/jobs/111",
                    title="Backend Engineer - CoolStartup | Wellfound",
                    snippet="30 LPA Remote India",
                ),
                SearchHit(
                    url="https://wellfound.com/jobs/222",
                    title="Software QA Intern - OtherCo",
                    snippet="Internship",
                ),
            ]

    res = await DorkDiscoverer(_FakeSerper(), repo).discover(pack, max_priority=2)
    # Priority 3 query skipped; suffix appended for serper; intern/qa title rejected
    assert res.queries_executed == 2
    assert all(q.endswith("-intern -fresher") for q in seen_queries)
    assert res.hits_rejected_by_title == 1
    assert len(res.alert_only_jobs) == 1
    assert res.alert_only_jobs[0].salary_raw == "LPA"


@respx.mock
async def test_serper_clamps_num_10_on_400_and_resolves_displayed_link_ats() -> None:
    seen_nums: list[int] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_nums.append(int(body["num"]))
        if int(body["num"]) > 10:
            return httpx.Response(
                400, json={"message": "Query pattern not allowed for free accounts."}
            )
        return httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "link": "https://www.google.com/goto?url=opaque_token",
                        "displayedLink": "https://jobs.ashbyhq.com › certifyos",
                        "title": "CertifyOS Jobs",
                        "snippet": "Remote India Backend Engineer",
                    },
                    {
                        "link": "https://www.google.com/goto?url=opaque_other",
                        "displayedLink": "https://randomblog.example.com › post",
                        "title": "Random blog",
                        "snippet": "Should be ignored as opaque google/goto wrapper",
                    },
                ]
            },
        )

    respx.post("https://google.serper.dev/search").mock(side_effect=_handler)
    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))

    async with httpx.AsyncClient() as client:
        provider = SerperSearchProvider("serper-test-key", client, limiter)
        hits = await provider.search('site:jobs.ashbyhq.com "India" -intern', max_results=20)
        # Subsequent call stays clamped to num=10 without hitting 400 again
        hits2 = await provider.search('site:jobs.lever.co "India" -intern', max_results=20)

    assert seen_nums == [20, 10, 10]
    assert len(hits) == 1
    assert hits[0].url == "https://jobs.ashbyhq.com/certifyos"
    assert len(hits2) == 1


@respx.mock
async def test_fallback_provider_strips_negations_for_ddg_and_disables_broken_primary() -> None:
    serper_route = respx.post("https://google.serper.dev/search").mock(
        return_value=httpx.Response(403, json={"message": "Forbidden"})
    )
    ddg_queries: list[str] = []

    def _fake_ddg(query: str, *, timelimit: str, max_results: int) -> list[dict[str, str]]:
        ddg_queries.append(query)
        return [
            {
                "href": "https://jobs.lever.co/acme-india/123",
                "title": "Backend Engineer",
                "body": "Remote India",
            }
        ]

    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    async with httpx.AsyncClient() as client:
        primary = SerperSearchProvider("bad-key", client, limiter)
        fallback = DuckDuckGoSearchProvider(limiter, search_fn=_fake_ddg)
        provider = FallbackSearchProvider(primary, fallback)

        hits1 = await provider.search(
            'site:jobs.lever.co India -intern -fresher -"0-1 years"', days=7, max_results=10
        )
        hits2 = await provider.search("site:jobs.ashbyhq.com India -intern", days=7, max_results=10)

    # 403 disabled primary after the first call; second query skipped primary completely
    assert serper_route.call_count == 1
    assert ddg_queries == [
        "site:jobs.lever.co India",
        "site:jobs.ashbyhq.com India",
    ]
    assert len(hits1) == 1
    assert len(hits2) == 1
