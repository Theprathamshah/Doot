# Doot

Local-first **Job Search Triage, RAG-Grounded Resume Tailoring, MCP Server, and Assisted-Apply Agent** built for macOS (Apple Silicon) with Python 3.11+ (`uv`).

## Documentation

- [REQUIREMENTS.md](REQUIREMENTS.md) — Scope, functional/non-functional requirements, and safety guardrails
- [HLD.md](HLD.md) — High-level architecture, Dork-to-Crawl loop, and SaaS evolution path
- [LLD.md](LLD.md) — SQLite WAL schema, enforced state machine, `Protocol` contracts, and config
- [PLAN.md](PLAN.md) — Phased implementation roadmap

## Phase 1–3: Quick Start

1. Copy `.env.example` to `.env` (`chmod 600 .env`), `data/profile.example.json` to `data/profile.json`, `data/resume.example.md` to `data/resume.md`, and `data/companies.example.yaml` to `data/companies.yaml`.
2. Add your `DOOT_GEMINI_API_KEY` to `.env`, then initialize, discover, and score:
   ```bash
   uv sync
   uv run doot init
   uv run doot sync-watchlist
   uv run doot add-company https://boards.greenhouse.io/postman --name Postman
   uv run doot discover --ats
   uv run doot score --stage1-only   # free rule filter only
   uv run doot score                 # Stage 1 + cached Gemini scoring (PII redacted)
   uv run doot stats
   ```
   Optional discovery modes:
   - Hybrid Dork-to-Crawl: `uv run doot discover --dork --ats` (requires `DOOT_SERPER_API_KEY` / `DOOT_BRAVE_API_KEY` / `DOOT_TAVILY_API_KEY`)
   - Logged-out JobSpy (requires `uv sync --extra jobspy`): `uv run doot discover --jobspy`
3. Run the offline test suite:
   ```bash
   uv run pytest
   ```

## Safety Invariants (Always Enforced)

- **Dry-run by default:** Submissions require explicit `--apply` **and** interactive confirmation.
- **Global kill switch:** Creating `data/STOP` aborts all write/apply operations immediately.
- **Volume caps:** Daily limit (`15`) and per-company limit (`2`) enforced in code before any application.
- **Audit trail:** Every job status transition is validated against `ALLOWED_TRANSITIONS` and logged in `job_events`.
