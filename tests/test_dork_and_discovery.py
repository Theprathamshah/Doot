"""Offline tests for SearchProviders, Dork-to-Crawl, auto-pause, and CLI discovery."""

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from doot.cli import app
from doot.collectors.dork import DorkDiscoverer, SerperSearchProvider
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
