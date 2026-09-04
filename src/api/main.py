"""FastAPI service: POST /query, POST /webhook, GET /health.

Long-lived components (search/rerank/LLM for answering; Confluence client +
indexer + page-state for the webhook sync path) are built once at startup and
held on app state.

Concurrency note: in Qdrant embedded mode (QDRANT_PATH set) the storage is a
SQLite that may only be used from the thread that created it. FastAPI runs sync
endpoints on a pool of worker threads, so we funnel every Qdrant-touching call
through a single dedicated worker thread (`_run`). With a real Qdrant server
(QDRANT_PATH empty) this is simply harmless serialization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from config import get_settings
from src.confluence.client import ConfluenceClient
from src.confluence.sync import PageStateStore, SyncStats, sync_page
from src.embeddings import Embedders
from src.generation.answer import Rag
from src.ingest.index import Indexer, _make_qdrant
from src.retrieval.rerank import Reranker
from src.retrieval.search import Searcher

T = TypeVar("T")


# --- request/response models ------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class CitationModel(BaseModel):
    title: str
    url: str
    last_modified: str
    page_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationModel]
    used_context: bool


# --- app container -----------------------------------------------------------

class Components:
    def __init__(self) -> None:
        self.settings = get_settings()
        # Single dedicated thread for all Qdrant-bound work (see module docstring).
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qdrant")
        self._run(self._construct)

    def _construct(self) -> None:
        s = self.settings
        # ONE Qdrant client + embedders shared by search and the webhook indexer.
        self.qdrant = _make_qdrant(s)
        self.embedders = Embedders(s)
        self.searcher = Searcher(s, qdrant=self.qdrant, embedders=self.embedders)
        self.reranker = Reranker(s)
        self.rag = Rag(s, searcher=self.searcher, reranker=self.reranker)
        self.indexer = Indexer(s, qdrant=self.qdrant, embedders=self.embedders)
        self.store = PageStateStore(s.page_state_db)
        self.confluence = ConfluenceClient(s.confluence_base_url, s.confluence_pat)

    def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute fn on the dedicated Qdrant thread and wait for the result."""
        return self._pool.submit(fn, *args, **kwargs).result()

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await run_in_threadpool(self._run, fn, *args, **kwargs)

    def close(self) -> None:
        def _shutdown() -> None:
            self.reranker.close()
            self.store.close()
            self.confluence.close()
            self.qdrant.close()

        self._run(_shutdown)
        self._pool.shutdown(wait=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.c = Components()
    try:
        yield
    finally:
        app.state.c.close()


app = FastAPI(title="Confluence RAG", version="0.1.0", lifespan=lifespan)


# --- endpoints ---------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")
    c: Components = request.app.state.c
    result = await c.run(c.rag.answer, question)
    return QueryResponse(**result.to_dict())


@app.get("/health")
async def health(request: Request):
    c: Components = request.app.state.c
    s = c.settings
    checks: Dict[str, Any] = {}

    def check_qdrant() -> str:
        try:
            c.qdrant.get_collections()
            return "ok"
        except Exception as exc:  # noqa: BLE001
            return f"error: {type(exc).__name__}"

    checks["qdrant"] = await c.run(check_qdrant)

    # Embeddings + LLM endpoints: cheap /models reachability probe.
    def probe(base_url: str, api_key: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            r = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=10.0)
            return "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            return f"error: {type(exc).__name__}"

    checks["embeddings"] = await run_in_threadpool(probe, s.embed_base_url, s.embed_api_key)
    checks["llm"] = await run_in_threadpool(probe, s.llm_base_url, s.llm_api_key)

    ok = all(v == "ok" for v in checks.values())
    body = {"status": "ok" if ok else "degraded", "checks": checks}
    if not ok:
        raise HTTPException(status_code=503, detail=body)
    return body


# Event names that mean the page is gone.
_DELETE_EVENTS = {"page_removed", "page_trashed", "page_deleted"}


def _extract_page_id(payload: Dict[str, Any]) -> Optional[str]:
    for path in (("page", "id"), ("content", "id"), ("page_id",), ("id",)):
        node: Any = payload
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if node is not None:
            return str(node)
    return None


@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    c: Components = request.app.state.c
    secret = x_webhook_secret or request.query_params.get("secret")
    if secret != c.settings.webhook_secret:
        raise HTTPException(status_code=401, detail="invalid webhook secret")

    payload = await request.json()
    event = str(payload.get("event") or payload.get("eventType") or "").lower()
    page_id = _extract_page_id(payload)
    if not page_id:
        raise HTTPException(status_code=422, detail="could not find page id in payload")

    # NOTE: M5 hardens this path (retry/backoff, structured logging, tests).
    def do_delete() -> dict:
        c.indexer.delete_page(page_id)
        c.store.delete(page_id)
        return {"status": "deleted", "page_id": page_id}

    def do_sync() -> dict:
        raw = c.confluence.get_page(page_id)
        stats = SyncStats()
        reason = sync_page(
            raw, client=c.confluence, store=c.store, stats=stats,
            on_index=c.indexer.index_page,
        )
        return {"status": "synced", "page_id": page_id, "result": reason or "indexed"}

    if any(e in event for e in _DELETE_EVENTS):
        return await c.run(do_delete)
    return await c.run(do_sync)
