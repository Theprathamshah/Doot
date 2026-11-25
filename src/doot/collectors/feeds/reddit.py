"""Reddit public JSON collector for genuine [Hiring] posts across dev subreddits."""

import re

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from doot.collectors.ats.detector import detect_ats_from_url
from doot.collectors.rate_limiter import DomainRateLimiter
from doot.models.job import RawJob

_DEFAULT_SUBREDDITS: tuple[str, ...] = (
    "forhire",
    "remotejs",
    "remotepython",
    "hiring",
    "developersIndia",
)

_SPAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcommission[\s-]+only\b", re.I),
    re.compile(r"\bunpaid\b", re.I),
    re.compile(r"\$\s*[1-9]\s*/\s*hr\b", re.I),
    re.compile(r"\bcrypto\s+airdrop\b", re.I),
    re.compile(r"\bonlyfans\b", re.I),
)

_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


class RedditHiringCollector:
    """Fetches [Hiring] posts from Reddit's public JSON endpoints (zero auth)."""

    source_name = "reddit"

    def __init__(
        self,
        client: httpx.AsyncClient,
        rate_limiter: DomainRateLimiter,
        *,
        subreddits: tuple[str, ...] = _DEFAULT_SUBREDDITS,
        limit_per_sub: int = 25,
    ) -> None:
        self.client = client
        self.rate_limiter = rate_limiter
        self.subreddits = subreddits
        self.limit_per_sub = limit_per_sub

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        reraise=True,
    )
    async def _fetch_sub(self, subreddit: str) -> list[dict[str, object]]:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={self.limit_per_sub}"
        await self.rate_limiter.acquire(url)
        resp = await self.client.get(
            url,
            headers={"User-Agent": "DootJobTriage/0.1 (local personal search)"},
            timeout=15.0,
        )
        if resp.status_code in (403, 404, 429):
            return []
        resp.raise_for_status()
        data = resp.json()
        children = (data.get("data") or {}).get("children") or []
        return [c.get("data") or {} for c in children if isinstance(c, dict)]

    async def fetch(self) -> list[RawJob]:
        """Collect verified [Hiring] posts while filtering out [For Hire] and spam."""
        raw_jobs: list[RawJob] = []
        seen_ids: set[str] = set()

        for sub in self.subreddits:
            try:
                posts = await self._fetch_sub(sub)
            except httpx.HTTPError:
                continue

            for post in posts:
                pid = str(post.get("id") or "").strip()
                title = str(post.get("title") or "").strip()
                selftext = str(post.get("selftext") or "").strip()
                permalink = str(post.get("permalink") or "").strip()
                if not pid or not title or pid in seen_ids:
                    continue

                title_low = title.lower()
                # Only ingest hiring posts, never [For Hire] self-promotion
                if "[for hire]" in title_low or "[forhire]" in title_low:
                    continue
                if "[hiring]" not in title_low and "hiring" not in title_low:
                    continue

                combined = f"{title}\n{selftext}"
                if any(pat.search(combined) for pat in _SPAM_PATTERNS):
                    continue

                seen_ids.add(pid)
                reddit_url = f"https://www.reddit.com{permalink}" if permalink else str(
                    post.get("url") or ""
                )
                urls = _URL_RE.findall(selftext)
                apply_url = urls[0] if urls else reddit_url

                detected = None
                for candidate_url in urls:
                    detected = detect_ats_from_url(candidate_url)
                    if detected is not None:
                        apply_url = detected.canonical_url or candidate_url
                        break

                is_remote = "remote" in combined.lower()
                is_india = "india" in combined.lower() or sub.lower() == "developersindia"
                loc = "Remote" if is_remote else ("India" if is_india else "")

                clean_title = re.sub(r"\[hiring\]\s*", "", title, flags=re.I).strip()
                raw_jobs.append(
                    RawJob(
                        source=f"reddit_{sub.lower()}",
                        external_job_id=pid,
                        title=clean_title[:160],
                        company_name=f"r/{sub}",
                        canonical_url=reddit_url,
                        apply_url=apply_url,
                        location_raw=loc,
                        jd_text=selftext or title,
                        ats_provider=detected.ats_provider if detected else None,
                        board_token=detected.board_token if detected else None,
                        alert_only=detected is None,
                    )
                )
        return raw_jobs
