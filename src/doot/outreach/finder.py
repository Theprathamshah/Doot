"""Public HR / Manager / Founder contact finder with MX check and pattern generation."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from doot.collectors.base import SearchHit, SearchProvider
from doot.config import DootSettings
from doot.db.repository import DootRepository

_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_IGNORED_EMAIL_PREFIXES = ("noreply", "no-reply", "donotreply", "support", "privacy", "legal")
_IGNORED_EMAIL_DOMAINS = frozenset(
    {
        "example.com",
        "sentry.io",
        "wixpress.com",
        "greenhouse.io",
        "lever.co",
        "ashbyhq.com",
        "linkedin.com",
        "wellfound.com",
    }
)
_AGGREGATOR_HOSTS = frozenset(
    {
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "jobs.lever.co",
        "jobs.ashbyhq.com",
        "www.linkedin.com",
        "linkedin.com",
        "wellfound.com",
        "www.wellfound.com",
        "jobicy.com",
        "remotive.com",
        "himalayas.app",
        "weworkremotely.com",
        "news.ycombinator.com",
        "www.reddit.com",
        "reddit.com",
        "x.com",
    }
)

MXCheckFn = Callable[[str], Awaitable[bool]]


@dataclass(slots=True)
class ContactDiscoveryResult:
    """Summary of contacts found and persisted for a target company/job."""

    company_name: str
    domain: str | None
    mx_valid: bool
    contact_ids: list[int] = field(default_factory=list)
    contacts_added: int = 0


def extract_emails_from_text(text: str) -> list[str]:
    """Extract plausible public hiring/contact emails from JD or snippet text."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _EMAIL_RE.findall(text):
        email = match.lower().rstrip(".")
        local, _, domain = email.partition("@")
        if local in _IGNORED_EMAIL_PREFIXES or domain in _IGNORED_EMAIL_DOMAINS:
            continue
        if email not in seen:
            seen.add(email)
            found.append(email)
    return found


def infer_company_domain(
    company_name: str,
    *,
    careers_url: str | None = None,
    jd_text: str = "",
) -> str | None:
    """Infer official company email domain from careers URL, JD emails/links, or fallback."""
    for email in extract_emails_from_text(jd_text):
        return email.partition("@")[2]

    if careers_url:
        host = (urlparse(careers_url).hostname or "").lower().removeprefix("www.")
        host = host.removeprefix("careers.").removeprefix("jobs.")
        if host and host not in _AGGREGATOR_HOSTS and "." in host:
            return host

    slug = re.sub(r"[^a-z0-9]", "", company_name.lower())
    return f"{slug}.com" if slug else None


def generate_email_candidates(person_name: str, domain: str) -> list[str]:
    """Generate ranked standard corporate email patterns for a person at a domain."""
    tokens = [re.sub(r"[^a-z]", "", p.lower()) for p in person_name.split()]
    tokens = [t for t in tokens if t]
    if not tokens or not domain:
        return []
    first = tokens[0]
    if len(tokens) == 1:
        return [f"{first}@{domain}"]
    last = tokens[-1]
    return [
        f"{first}.{last}@{domain}",
        f"{first}@{domain}",
        f"{first[0]}{last}@{domain}",
        f"{first[0]}.{last}@{domain}",
    ]


def parse_linkedin_people_hits(hits: list[SearchHit]) -> list[tuple[str, str, str]]:
    """Parse `(person_name, role_title, linkedin_url)` from public LinkedIn search hits."""
    people: list[tuple[str, str, str]] = []
    seen_names: set[str] = set()
    for hit in hits:
        if "linkedin.com/in/" not in hit.url.lower():
            continue
        cleaned = re.sub(r"\s*\|\s*LinkedIn.*$", "", hit.title, flags=re.I).strip()
        parts = [p.strip() for p in re.split(r"\s+[-–|]\s+", cleaned) if p.strip()]
        if not parts:
            continue
        name = parts[0]
        if len(name.split()) < 2 or len(name) > 50 or name.lower() in seen_names:
            continue
        role = parts[1] if len(parts) >= 2 else "Hiring / Engineering Leader"
        seen_names.add(name.lower())
        people.append((name, role[:120], hit.url))
    return people


async def check_domain_has_mx(domain: str) -> bool:
    """Verify via DNS MX lookup (`nslookup`) that the domain accepts emails."""
    if not domain or "." not in domain:
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "nslookup",
            "-type=MX",
            domain,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        text = stdout.decode("utf-8", "ignore").lower()
        return "mail exchanger" in text or "mx" in text
    except Exception:
        return False


async def _fetch_hunter_contacts(
    client: httpx.AsyncClient,
    domain: str,
    api_key: str,
) -> list[tuple[str, str, str]]:
    """Query Hunter.io domain-search if DOOT_HUNTER_API_KEY is configured."""
    if not api_key or not domain:
        return []
    resp = await client.get(
        "https://api.hunter.io/v2/domain-search",
        params={"domain": domain, "api_key": api_key, "limit": 10},
        timeout=15.0,
    )
    if resp.status_code != 200:
        return []
    data = resp.json()
    emails = (data.get("data") or {}).get("emails") or []
    results: list[tuple[str, str, str]] = []
    for item in emails:
        addr = str(item.get("value") or "").strip().lower()
        if not addr:
            continue
        first = str(item.get("first_name") or "").strip()
        last = str(item.get("last_name") or "").strip()
        name = f"{first} {last}".strip()
        role = str(item.get("position") or "").strip()
        results.append((name, role, addr))
    return results


async def find_contacts_for_company(
    settings: DootSettings,
    repo: DootRepository,
    *,
    company_name: str,
    domain: str | None = None,
    careers_url: str | None = None,
    jd_text: str = "",
    company_id: int | None = None,
    job_id: int | None = None,
    search_provider: SearchProvider | None = None,
    client: httpx.AsyncClient | None = None,
    mx_check_fn: MXCheckFn | None = None,
) -> ContactDiscoveryResult:
    """Discover public HR, Engineering Manager, and Founder contacts safely."""
    resolved_domain = domain or infer_company_domain(
        company_name, careers_url=careers_url, jd_text=jd_text
    )
    mx_fn = mx_check_fn or check_domain_has_mx
    mx_valid = await mx_fn(resolved_domain) if resolved_domain else False

    result = ContactDiscoveryResult(
        company_name=company_name,
        domain=resolved_domain,
        mx_valid=mx_valid,
    )

    async def _save(
        email: str,
        *,
        person_name: str = "",
        role_title: str = "",
        confidence: str = "pattern",
        source: str = "dork",
        linkedin_url: str | None = None,
    ) -> None:
        cid = await repo.upsert_contact(
            company_name=company_name,
            email=email,
            person_name=person_name,
            role_title=role_title,
            confidence=confidence,
            source=source,
            linkedin_url=linkedin_url,
            company_id=company_id,
            job_id=job_id,
        )
        if cid not in result.contact_ids:
            result.contact_ids.append(cid)
            result.contacts_added += 1

    # 1) Direct emails mentioned in JD / HN / Reddit text
    for email in extract_emails_from_text(jd_text):
        await _save(email, role_title="Hiring Contact (JD)", confidence="verified_jd", source="jd")

    # 2) Optional Hunter.io verified lookup if API key is provided
    hunter_key = settings.hunter_api_key.get_secret_value()
    if hunter_key and resolved_domain and client is not None:
        for name, role, addr in await _fetch_hunter_contacts(client, resolved_domain, hunter_key):
            await _save(
                addr,
                person_name=name,
                role_title=role,
                confidence="hunter_verified",
                source="hunter",
            )

    # 3) Search dorking for public leadership/recruiter profiles + public hiring emails
    if search_provider is not None:
        if resolved_domain:
            email_hits = await search_provider.search(
                f'site:{resolved_domain} ("careers@" OR "jobs@" OR "hiring@" OR "talent@")',
                days=365,
                max_results=10,
            )
            for hit in email_hits:
                for addr in extract_emails_from_text(f"{hit.title} {hit.snippet}"):
                    await _save(
                        addr,
                        role_title="Public Hiring Alias",
                        confidence="public_search",
                        source=search_provider.provider_name,
                    )

        people_hits = await search_provider.search(
            f'site:linkedin.com/in ("Engineering Manager" OR "Tech Recruiter" OR '
            f'"Talent Acquisition" OR "CTO" OR "Founder") "{company_name}"',
            days=365,
            max_results=10,
        )
        for name, role, li_url in parse_linkedin_people_hits(people_hits):
            if resolved_domain:
                candidates = generate_email_candidates(name, resolved_domain)
                if candidates:
                    await _save(
                        candidates[0],
                        person_name=name,
                        role_title=role,
                        confidence="pattern_mx" if mx_valid else "pattern",
                        source=search_provider.provider_name,
                        linkedin_url=li_url,
                    )

    # 4) Role-alias fallbacks when MX record confirms the domain receives mail
    if resolved_domain and mx_valid and result.contacts_added == 0:
        for alias in ("careers", "jobs", "talent"):
            await _save(
                f"{alias}@{resolved_domain}",
                role_title="Recruiting Inbox (MX-verified domain)",
                confidence="role_alias_mx",
                source="mx_pattern",
            )

    return result
