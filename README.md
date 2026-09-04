# Confluence RAG

Retrieval-augmented Q&A over a self-hosted **Confluence Data Center** instance,
using OpenAI-compatible endpoints for generation and embeddings. See
[`plan.md`](plan.md) for the full spec and milestone plan.

## Status

- **M1 — Extraction + sync foundation** ✅
  Confluence REST client (PAT auth, pagination, CQL delta, read-restriction
  lookup), page extraction + metadata + content hashing, SQLite page-state
  store, and the v1 safety rule (space allowlist + skip restricted pages).
- M2–M5: not started.

## Setup

Requires Python 3.9+ (the plan targets 3.11+; only 3.9 is available on this
host, and the code is kept compatible with both).

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env      # then fill in real values
```

Configuration is entirely environment-driven (`.env`, never committed). The
Confluence PAT may be supplied as `CONFLUENCE_PAT` or `CONFLUENCE_TOKEN`.

## Run (M1)

```bash
# Full crawl of the allowlisted spaces (CONFLUENCE_SPACES)
./.venv/bin/python -m scripts.full_sync

# Delta re-pull of pages changed since a timestamp
./.venv/bin/python -m scripts.full_sync --since "2026-09-01 00:00"
```

The script prints counts: `pages seen / indexed / skipped-restricted /
skipped-unchanged`. A re-run re-pulls only pages whose content hash changed.

## Test

```bash
./.venv/bin/python -m pytest
```
