"""Cross-encoder reranking over hybrid-retrieval candidates.

Calls the BGE-reranker-v2-m3 endpoint (Cohere/Jina-style /rerank: a JSON body of
{model, query, documents} returning {results: [{index, relevance_score}]}),
keeps the top RERANK_TOP_N, and drops everything below RERANK_SCORE_MIN. If no
candidate clears the threshold, returns an empty list — the M4 "no relevant
context" signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx

from config import Settings, get_settings
from src.retrieval.search import Candidate


@dataclass
class Reranked:
    chunk_id: str
    page_id: str
    score: float  # cross-encoder relevance in [0, 1]
    payload: dict

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


class Reranker:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.settings = settings or get_settings()
        headers = {}
        if self.settings.rerank_api_key:
            headers["Authorization"] = f"Bearer {self.settings.rerank_api_key}"
        self._client = client or httpx.Client(headers=headers, timeout=30.0)

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        *,
        top_n: Optional[int] = None,
        score_min: Optional[float] = None,
    ) -> List[Reranked]:
        top_n = top_n or self.settings.rerank_top_n
        score_min = self.settings.rerank_score_min if score_min is None else score_min
        if not candidates:
            return []

        documents = [c.text for c in candidates]
        resp = self._client.post(
            self.settings.rerank_url,
            json={
                "model": self.settings.rerank_model,
                "query": query,
                "documents": documents,
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        # Sort by relevance desc, keep those clearing the threshold, cap at top_n.
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        out: List[Reranked] = []
        for r in results:
            score = float(r["relevance_score"])
            if score < score_min:
                continue
            cand = candidates[int(r["index"])]
            out.append(
                Reranked(
                    chunk_id=cand.chunk_id,
                    page_id=cand.page_id,
                    score=score,
                    payload=cand.payload,
                )
            )
            if len(out) >= top_n:
                break
        return out

    def close(self) -> None:
        self._client.close()
