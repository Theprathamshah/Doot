# Doot

Local-first **Job Search Triage, RAG-Grounded Resume Tailoring, MCP Server, and Assisted-Apply Agent** built for macOS (Apple Silicon) with Python 3.11+ (`uv`).

## Documentation

- [REQUIREMENTS.md](REQUIREMENTS.md) — Scope, functional/non-functional requirements, and safety guardrails
- [HLD.md](HLD.md) — High-level architecture, Dork-to-Crawl loop, and SaaS evolution path
- [LLD.md](LLD.md) — SQLite WAL schema, enforced state machine, `Protocol` contracts, and config
- [PLAN.md](PLAN.md) — Phased implementation roadmap

## Phase 1: Quick Start

1. Copy `.env.example` to `.env` (`chmod 600 .env`) and copy `data/profile.example.json` to `data/profile.json`.
2. Initialize the database and verify setup:
   ```bash
   uv sync
   uv run doot init
   uv run doot status
   uv run doot stats
   ```
3. Run the offline test suite:
   ```bash
   uv run pytest
   ```

## Safety Invariants (Always Enforced)

- **Dry-run by default:** Submissions require explicit `--apply` **and** interactive confirmation.
- **Global kill switch:** Creating `data/STOP` aborts all write/apply operations immediately.
- **Volume caps:** Daily limit (`15`) and per-company limit (`2`) enforced in code before any application.
- **Audit trail:** Every job status transition is validated against `ALLOWED_TRANSITIONS` and logged in `job_events`.
