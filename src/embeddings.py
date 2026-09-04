"""Shared embedding helpers: dense (OpenAI-compatible endpoint) + BM25 sparse
(fastembed). Used by both indexing (ingest/index.py) and query-time retrieval
(retrieval/search.py) so the same vectors are produced on both sides.
"""

from __future__ import annotations

from typing import List, Optional

from fastembed import SparseTextEmbedding
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from qdrant_client import models

from config import Settings, get_settings

_BM25_MODEL = "Qdrant/bm25"


class Embedders:
    """Dense + sparse embedders bound to the configured endpoint/model."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = OpenAI(
            base_url=self.settings.embed_base_url,
            api_key=self.settings.embed_api_key,
        )
        self._bm25 = SparseTextEmbedding(model_name=_BM25_MODEL)

    def dense(self, texts: List[str]) -> List[List[float]]:
        # NOTE: low-level .post() rather than embeddings.create() — the SDK's
        # convenience method auto-injects encoding_format=base64, which the
        # LiteLLM gateway rejects for the qwen3-embedding model group. Building
        # the body ourselves omits it and the server returns float arrays.
        resp = self._client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={"model": self.settings.embed_model, "input": texts},
        )
        return [d.embedding for d in resp.data]

    def dense_one(self, text: str) -> List[float]:
        return self.dense([text])[0]

    def sparse(self, texts: List[str]) -> List[models.SparseVector]:
        out: List[models.SparseVector] = []
        for emb in self._bm25.embed(texts):
            out.append(
                models.SparseVector(
                    indices=emb.indices.tolist(), values=emb.values.tolist()
                )
            )
        return out

    def sparse_one(self, text: str) -> models.SparseVector:
        # query_embed() applies BM25 query-side weighting (no term frequencies).
        emb = next(iter(self._bm25.query_embed(text)))
        return models.SparseVector(
            indices=emb.indices.tolist(), values=emb.values.tolist()
        )
