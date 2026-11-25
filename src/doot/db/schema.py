"""SQLite DDL and connection pragmas."""

import aiosqlite

PRAGMAS_SQL: tuple[str, ...] = (
    "PRAGMA journal_mode = WAL;",
    "PRAGMA foreign_keys = ON;",
    "PRAGMA busy_timeout = 5000;",
    "PRAGMA synchronous = NORMAL;",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    ats_provider TEXT NOT NULL CHECK (ats_provider IN ('greenhouse', 'lever', 'ashby', 'custom')),
    board_token TEXT NOT NULL,
    careers_url TEXT,
    source TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'dork', 'seed')),
    is_blocklisted INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_crawled_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (ats_provider, board_token)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER REFERENCES companies(id) ON DELETE SET NULL,
    source TEXT NOT NULL,
    external_job_id TEXT NOT NULL,
    ats_provider TEXT,
    board_token TEXT,
    canonical_url TEXT NOT NULL,
    apply_url TEXT NOT NULL,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    company_name TEXT NOT NULL,
    normalized_company TEXT NOT NULL,
    location_raw TEXT NOT NULL DEFAULT '',
    is_remote INTEGER NOT NULL DEFAULT 0,
    india_eligible INTEGER NOT NULL DEFAULT 1,
    alert_only INTEGER NOT NULL DEFAULT 0,
    salary_raw TEXT,
    jd_text TEXT NOT NULL DEFAULT '',
    jd_hash TEXT NOT NULL,
    dedupe_cluster_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'DISCOVERED',
    rule_filter_reason TEXT,
    llm_score INTEGER CHECK (llm_score BETWEEN 1 AND 5),
    llm_reasoning TEXT,
    missing_skills_json TEXT,
    should_apply INTEGER,
    score_cache_key TEXT,
    discovered_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (source, external_job_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_ats_key ON jobs(ats_provider, board_token, external_job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_score_cache ON jobs(score_cache_key);
CREATE INDEX IF NOT EXISTS idx_jobs_dedupe_cluster ON jobs(dedupe_cluster_id);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    from_status TEXT,
    to_status TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'PREPARED',
    tailored_summary TEXT,
    cover_note TEXT,
    outreach_note TEXT,
    resume_path TEXT,
    filled_fields_json TEXT,
    screenshot_path TEXT,
    error_message TEXT,
    submitted_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS qa_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_normalized TEXT NOT NULL,
    question_sample TEXT NOT NULL,
    answer TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global' CHECK (scope IN ('global', 'company')),
    company_id INTEGER REFERENCES companies(id) ON DELETE CASCADE,
    confirmed_by_user INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_qa_cache_lookup
    ON qa_cache(question_normalized, scope, COALESCE(company_id, -1));

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at TEXT,
    stats_json TEXT,
    error_message TEXT
);
"""


async def apply_pragmas(conn: aiosqlite.Connection) -> None:
    """Apply required SQLite WAL, foreign-key, and concurrency pragmas."""
    for pragma in PRAGMAS_SQL:
        await conn.execute(pragma)


async def initialize_schema(conn: aiosqlite.Connection) -> None:
    """Create all Doot tables and indexes idempotently."""
    await apply_pragmas(conn)
    await conn.executescript(SCHEMA_SQL)
    await conn.commit()
