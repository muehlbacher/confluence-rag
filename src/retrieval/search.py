"""Hybrid retrieval: dense + BM25 over Qdrant, fused with Reciprocal Rank Fusion.

Uses Qdrant's Query API: two prefetches (dense vector search + BM25 sparse
search), each to RETRIEVE_TOP_K, fused server-side with RRF. Returns candidate
chunks with their payloads for reranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from qdrant_client import QdrantClient, models

from config import Settings, get_settings
from src.embeddings import Embedders
from src.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, _make_qdrant


@dataclass
class Candidate:
    chunk_id: str
    page_id: str
    score: float  # fusion score (RRF); not comparable to rerank scores
    payload: dict

    @property
    def text(self) -> str:
        return self.payload.get("text", "")


class Searcher:
    """Hybrid dense+BM25 retriever against the Qdrant collection."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        qdrant: Optional[QdrantClient] = None,
        embedders: Optional[Embedders] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self.qdrant = qdrant or _make_qdrant(self.settings)
        self.embedders = embedders or Embedders(self.settings)

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        space_keys: Optional[List[str]] = None,
    ) -> List[Candidate]:
        """Return up to top_k fused candidates for the query."""
        top_k = top_k or self.settings.retrieve_top_k

        dense_vec = self.embedders.dense_one(query)
        sparse_vec = self.embedders.sparse_one(query)

        query_filter = None
        if space_keys:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="space_key", match=models.MatchAny(any=list(space_keys))
                    )
                ]
            )

        result = self.qdrant.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense_vec, using=DENSE_VECTOR, limit=top_k),
                models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR, limit=top_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        candidates: List[Candidate] = []
        for point in result.points:
            payload = point.payload or {}
            candidates.append(
                Candidate(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    page_id=payload.get("page_id", ""),
                    score=point.score,
                    payload=payload,
                )
            )
        return candidates

    def close(self) -> None:
        self.qdrant.close()
