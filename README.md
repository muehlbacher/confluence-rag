# Confluence RAG

Retrieval-augmented Q&A over a self-hosted **Confluence Data Center** instance,
using OpenAI-compatible endpoints for generation and embeddings. See
[`plan.md`](plan.md) for the full spec and milestone plan.

## Status

- **M1 — Extraction + sync foundation** ✅
  Confluence REST client (PAT auth, pagination, CQL delta, read-restriction
  lookup), page extraction + metadata + content hashing, SQLite page-state
  store, and the v1 safety rule (space allowlist + skip restricted pages).
- **M2 — Chunk + embed + index** ✅
  `body.view` → clean typed blocks (parse), heading-aware breadcrumb chunking
  (tables/code kept whole, split only above the embedder's context limit),
  dense embeddings via the OpenAI-compatible endpoint + BM25 sparse vectors,
  upserted to Qdrant (named `dense` + `bm25` vectors, per-page replace).
- **M3 — Hybrid retrieval + rerank + eval** ✅
  Dense + BM25 over Qdrant fused with RRF (Query API), cross-encoder reranking
  (BGE-reranker-v2-m3) to `RERANK_TOP_N` with a `RERANK_SCORE_MIN` gate, and an
  eval harness (hit-rate@k / MRR) over `eval/dataset.jsonl`.
- **M4 — Generation with citations + API** ✅
  `Rag` pipeline (search → rerank → LLM) grounded strictly in retrieved context,
  with inline `[n]` citations resolved to real pages, a two-stage no-context
  gate (empty retrieval *and* model-side refusal), and prompt-injection defense.
  FastAPI: `POST /query`, `POST /webhook`, `GET /health`.
- **M5 — Sync loop + hardening** ✅
  Webhook wired into the sync path (create/update/delete propagation, verified
  live end-to-end), `scripts/reindex.py`, retry/backoff on Confluence + reranker
  (and the OpenAI SDK's own retries for embeddings/LLM), structured logging, and
  retrieval tuning against the eval set.

### Retrieval eval (final, 20-question German set)

| config | hit-rate@5 | MRR |
|--------|-----------|-----|
| reranker OFF (RRF fusion) | 0.950 | 0.703 |
| reranker ON | **1.000** | **0.818** |

Config that produced these numbers: embeddings `qwen3-embedding-0.6b` (1024-dim)
dense + Qdrant BM25 sparse, RRF fusion, reranker `bge-reranker-v2-m3`;
`RETRIEVE_TOP_K=12`, `RERANK_TOP_N=5`, `RERANK_SCORE_MIN=0.05`,
`CHUNK_MAX_TOKENS=512`. Reranking lifts MRR ~16% relative and closes the last
top-5 miss. Tuning note: `RETRIEVE_TOP_K` was swept 6–40; 10–12 is the sweet
spot (fewer distractors for the reranker). Run: `./.venv/bin/python -m eval.run_eval`.

### Deviations from the plan (driven by the live environment)

- **Models:** the plan assumes BGE-M3 embeddings and a Qwen2.5 LLM; the endpoint
  offers neither. Using **`qwen3-embedding-0.6b`** (1024-dim, `EMBED_DIM=1024`)
  and **`qwen3.6-35b-a3b`** for generation. The embedding model is reached via
  the OpenAI client's low-level `.post()` because the gateway rejects the
  `encoding_format` param that `embeddings.create()` auto-injects (see
  `src/ingest/index.py`).
- **Qdrant:** no Docker on this host, so local dev uses qdrant-client's embedded
  mode via `QDRANT_PATH` (e.g. `qdrant_storage`). `docker-compose.yml` is still
  provided for server deployment — leave `QDRANT_PATH` empty to use it.
- **Reasoning LLM:** `qwen3.6-35b-a3b` spends the token budget "thinking"
  before answering (a small `max_tokens` yields empty content). Generation
  disables it via `chat_template_kwargs={"enable_thinking": false}`
  (`LLM_DISABLE_THINKING`).
- **Rerank threshold:** the plan's `RERANK_SCORE_MIN=0.3` was too high for
  bge-reranker-v2-m3 on this corpus (in-corpus top-1 scores start ~0.145,
  out-of-corpus ≤0.008). Calibrated to `0.05`; M5 formalizes the tuning.
- **Python:** 3.9 (plan targets 3.11+); code is kept compatible.

## Setup

Requires Python 3.9+ (the plan targets 3.11+; only 3.9 is available on this
host, and the code is kept compatible with both).

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev,index]"
cp .env.example .env      # then fill in real values
```

Configuration is entirely environment-driven (`.env`, never committed). The
Confluence PAT may be supplied as `CONFLUENCE_PAT` or `CONFLUENCE_TOKEN`.

## Run

```bash
# Full crawl + index of the allowlisted spaces (CONFLUENCE_SPACES)
./.venv/bin/python -m scripts.full_sync

# Drop the Qdrant collection + page-state, then rebuild from scratch
./.venv/bin/python -m scripts.full_sync --recreate

# Delta re-pull of pages changed since a timestamp
./.venv/bin/python -m scripts.full_sync --since "2026-09-01 00:00"

# Extraction only, no embeddings / Qdrant (M1 behaviour)
./.venv/bin/python -m scripts.full_sync --no-index

# Destructive drop + full rebuild (after model/dim/chunking changes)
./.venv/bin/python -m scripts.reindex
```

The script prints counts (`seen / indexed / skipped-restricted /
skipped-unchanged`) and the resulting Qdrant point count. A re-run re-pulls and
re-embeds only pages whose content hash changed; an unchanged corpus is a no-op.

## Serve the API

```bash
./.venv/bin/python -m pip install -e ".[api]"
./.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- `POST /query` — `{"question": "..."}` → `{answer, citations[], used_context}`.
  ```bash
  curl -s localhost:8000/query -H 'content-type: application/json' \
    -d '{"question":"Wie erstelle ich einen S3 Bucket?"}'
  ```
- `GET /health` — reachability of Qdrant, the embeddings endpoint, and the LLM.
- `POST /webhook` — secret-verified (header `X-Webhook-Secret` or `?secret=`);
  syncs/deletes the affected page (hardened in M5).

Note: in Qdrant **embedded** mode (`QDRANT_PATH` set) all vector-store access is
funnelled through a single thread, since embedded storage is single-threaded.
For real deployment run Qdrant as a server (leave `QDRANT_PATH` empty; use
`docker-compose.yml`).

## Test

```bash
./.venv/bin/python -m pytest
```
