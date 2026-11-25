"""Offline tests for public remote/startup feeds and DuckDuckGo SearchProvider."""

import httpx
import respx

from doot.collectors.dork import DorkDiscoverer, DuckDuckGoSearchProvider
from doot.collectors.feeds import (
    HimalayasCollector,
    HNHiringCollector,
    JobicyCollector,
    RemotiveCollector,
    WeWorkRemotelyCollector,
)
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.config import DootSettings
from doot.db import DootRepository
from doot.models import ATSProvider
from doot.pipeline import run_discovery_pipeline


@respx.mock
async def test_public_remote_and_hn_feeds(
    tmp_settings: DootSettings,
    repo: DootRepository,
) -> None:
    respx.get("https://himalayas.app/jobs/api?limit=20&offset=0").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "guid": "https://himalayas.app/companies/acme/jobs/backend-1",
                        "applicationLink": "https://himalayas.app/jobs/backend-1/apply",
                        "title": "Senior Backend Engineer (AWS)",
                        "companyName": "CloudCo",
                        "locationRestrictions": [],
                        "minSalary": 60000,
                        "maxSalary": 90000,
                        "currency": "USD",
                        "description": "<p>AWS Lambda and Node.js backend.</p>",
                    }
                ]
            },
        )
    )
    respx.get("https://remotive.com/api/remote-jobs?category=software-dev&limit=10").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 771,
                        "url": "https://remotive.com/remote-jobs/software-dev/fullstack-771",
                        "title": "Full Stack Engineer (React + Java)",
                        "company_name": "RemotePay",
                        "candidate_required_location": "Worldwide",
                        "salary": "$70k-$95k",
                        "description": "<p>Spring Boot and React.</p>",
                    }
                ]
            },
        )
    )
    respx.get("https://jobicy.com/api/v2/remote-jobs?count=10&industry=dev").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 882,
                        "url": "https://jobicy.com/jobs/882",
                        "jobTitle": "AI Engineer (RAG / MCP)",
                        "companyName": "VectorLabs",
                        "jobGeo": "Anywhere",
                        "salaryMin": 80000,
                        "salaryMax": 110000,
                        "salaryCurrency": "USD",
                        "jobDescription": "<p>Python, LLM, RAG.</p>",
                    }
                ]
            },
        )
    )
    respx.get(
        "https://hn.algolia.com/api/v1/search_by_date"
        "?query=%22Who%20is%20hiring%22&tags=story,author_whoishiring&hitsPerPage=1"
    ).mock(return_value=httpx.Response(200, json={"hits": [{"objectID": "44001"}]}))
    respx.get("https://hn.algolia.com/api/v1/items/44001").mock(
        return_value=httpx.Response(
            200,
            json={
                "children": [
                    {
                        "id": 44099,
                        "text": (
                            'StartupY | Backend Engineer | Remote (India/APAC) | '
                            '<a href="https://jobs.ashbyhq.com/startupy/abc-123-full-uuid">'
                            "https://jobs.ashbyhq.com/startupy/abc-123...</a>"
                        ),
                    }
                ]
            },
        )
    )

    respx.get("https://weworkremotely.com/categories/remote-programming-jobs.rss").mock(
        return_value=httpx.Response(
            200,
            text="""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Acme Labs: Backend Engineer (AWS/Node)</title>
      <link>https://weworkremotely.com/remote-jobs/acme-backend</link>
      <region>Anywhere in the World</region>
      <description>&lt;p&gt;AWS Lambda and Node.js.&lt;/p&gt;</description>
    </item>
  </channel>
</rss>""",
        )
    )

    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    async with httpx.AsyncClient() as client:
        summary = await run_discovery_pipeline(
            tmp_settings,
            repo,
            client=client,
            rate_limiter=limiter,
            use_ats=False,
            use_feeds=True,
            feed_collectors=[
                HimalayasCollector(client, limiter, limit=10),
                RemotiveCollector(client, limiter, limit=10),
                JobicyCollector(client, limiter, count=10),
                HNHiringCollector(client, limiter, limit=10),
                WeWorkRemotelyCollector(client, limiter, limit=10),
            ],
        )

    assert summary.jobs_created == 5
    jobs = await repo.list_jobs()
    by_source = {j.source: j for j in jobs}
    assert by_source["himalayas"].india_eligible is True
    assert by_source["remotive"].salary_raw == "$70k-$95k"
    assert by_source["jobicy"].is_remote is True
    assert by_source["weworkremotely"].company_name == "Acme Labs"
    assert by_source["weworkremotely"].india_eligible is True
    assert by_source["hn_hiring"].ats_provider == ATSProvider.ASHBY
    assert by_source["hn_hiring"].board_token == "startupy"
    assert (
        by_source["hn_hiring"].apply_url
        == "https://jobs.ashbyhq.com/startupy/abc-123-full-uuid"
    )


async def test_ddg_search_provider_and_wellfound_linkedin_title_inference(
    repo: DootRepository,
) -> None:
    limiter = DomainRateLimiter(min_interval_seconds=0.0, jitter_range=(0.0, 0.0))
    ddg = DuckDuckGoSearchProvider(
        limiter,
        search_fn=lambda q, **_: [
            {
                "href": "https://www.linkedin.com/jobs/view/123456",
                "title": "Senior Backend Engineer - FinStack | LinkedIn",
                "body": "Remote India AWS Java Spring Boot",
            },
            {
                "href": "https://wellfound.com/jobs/999-ai-eng",
                "title": "AI Engineer at PromptWorks | Wellfound",
                "body": "India Remote LLM RAG MCP",
            },
        ],
    )
    res = await DorkDiscoverer(ddg, repo).discover(["query"])
    assert len(res.alert_only_jobs) == 2
    assert res.alert_only_jobs[0].title == "Senior Backend Engineer"
    assert res.alert_only_jobs[0].company_name == "FinStack"
    assert res.alert_only_jobs[0].location_raw == "Remote (India)"
    assert res.alert_only_jobs[1].title == "AI Engineer"
    assert res.alert_only_jobs[1].company_name == "PromptWorks"
