"""Normalization, canonical URL cleanup, and deduplication helpers."""

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from doot.collectors.ats.detector import detect_ats_from_url
from doot.models.job import JobPosting, RawJob

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ref",
        "source",
        "gh_src",
        "lever-source",
        "lever-origin",
        "ashby_jid",
    }
)

_CITY_ALIASES: dict[str, str] = {
    "bangalore": "bengaluru",
    "gurgaon": "gurugram",
    "bombay": "mumbai",
    "madras": "chennai",
}

_INDIA_HINTS = (
    "india",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "pune",
    "mumbai",
    "delhi",
    "ncr",
    "gurugram",
    "gurgaon",
    "noida",
    "chennai",
    "ahmedabad",
    "kolkata",
    "apac",
    "anywhere",
    "worldwide",
    "global",
    "all regions",
)

_US_ONLY_HINTS = (
    "us only",
    "u.s. only",
    "usa only",
    "united states only",
    "north america only",
    "americas only",
    "latam only",
    "remote - us",
    "remote (us)",
    "must be authorized to work in the united states",
    "must reside in the us",
    "uk only",
    "eu only",
    "europe only",
    "emea only",
    "canada only",
)


def normalize_text(value: str) -> str:
    """Lowercase, strip punctuation, normalize city aliases, and collapse spaces."""
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    tokens = [_CITY_ALIASES.get(tok, tok) for tok in cleaned.split()]
    return " ".join(tokens)


def canonicalize_url(url: str) -> str:
    """Strip common tracking query parameters and normalize trailing slashes."""
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip()
    filtered_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            urlencode(filtered_query),
            "",
        )
    )


def compute_jd_hash(jd_text: str) -> str:
    """Compute stable SHA-256 hash over normalized job description text."""
    normalized = re.sub(r"\s+", " ", jd_text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_dedupe_cluster_id(
    normalized_company: str,
    normalized_title: str,
    jd_hash: str,
) -> str:
    """Build a clustering key for flagging probable cross-source duplicates."""
    material = f"{normalized_company}|{normalized_title}|{jd_hash[:12]}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


_BROAD_REMOTE_MARKERS = (
    "india",
    "apac",
    "anywhere",
    "worldwide",
    "global",
    "all regions",
    "earth",
    "work from anywhere",
)


def infer_remote_and_india(location_raw: str) -> tuple[bool, bool]:
    """Infer `(is_remote, india_eligible)` flags from location metadata."""
    loc_lower = location_raw.strip().lower()
    is_remote = "remote" in loc_lower or "work from home" in loc_lower or "wfh" in loc_lower

    if any(hint in loc_lower for hint in _US_ONLY_HINTS):
        return is_remote, False
    if any(hint in loc_lower for hint in _INDIA_HINTS):
        return is_remote, True
    if is_remote:
        # Plain "Remote" / "Remote - Full Time" passes; "Remote (Canada)" / "Remote (China)" fails
        stripped = re.sub(r"\b(?:remote|work from home|wfh|hybrid|full[- ]time)\b", "", loc_lower)
        stripped = re.sub(r"[^a-z]+", "", stripped)
        if not stripped:
            return True, True
        return True, any(marker in loc_lower for marker in _BROAD_REMOTE_MARKERS)
    if not loc_lower:
        return False, True
    # Explicit non-India on-site city (e.g. "San Francisco, CA")
    return False, False


def normalize_raw_job(raw: RawJob, *, company_id: int | None = None) -> JobPosting:
    """Convert an unnormalized RawJob into a canonical JobPosting."""
    canonical_url = canonicalize_url(raw.canonical_url)
    apply_url = raw.apply_url.strip() or canonical_url
    detected = detect_ats_from_url(canonical_url) or detect_ats_from_url(apply_url)

    ats_provider = raw.ats_provider or (detected.ats_provider if detected else None)
    board_token = raw.board_token or (detected.board_token if detected else None)

    norm_company = normalize_text(raw.company_name)
    norm_title = normalize_text(raw.title)
    jd_hash = compute_jd_hash(raw.jd_text)
    cluster_id = compute_dedupe_cluster_id(norm_company, norm_title, jd_hash)
    is_remote, india_eligible = infer_remote_and_india(raw.location_raw)

    return JobPosting(
        company_id=company_id,
        source=raw.source,
        external_job_id=raw.external_job_id,
        ats_provider=ats_provider,
        board_token=board_token,
        canonical_url=canonical_url,
        apply_url=apply_url,
        title=raw.title.strip(),
        normalized_title=norm_title,
        company_name=raw.company_name.strip(),
        normalized_company=norm_company,
        location_raw=raw.location_raw.strip(),
        is_remote=is_remote,
        india_eligible=india_eligible,
        alert_only=raw.alert_only,
        salary_raw=raw.salary_raw,
        jd_text=raw.jd_text.strip(),
        jd_hash=jd_hash,
        dedupe_cluster_id=cluster_id,
    )
