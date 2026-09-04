"""Embed chunks (dense via OpenAI-compatible endpoint) and upsert dense + BM25
sparse vectors into Qdrant.

Per the plan's embeddings note: the OpenAI `/v1/embeddings` contract is dense
only, so the sparse half of hybrid retrieval comes from Qdrant's BM25. We
compute BM25 term-frequency sparse vectors client-side with fastembed and set
the collection's sparse modifier to IDF so Qdrant scores them as BM25 at query
time.

# NOTE: To switch to BGE-M3's *learned* sparse vectors later, replace the
# fastembed BM25 here with a dense+sparse embedder call — this is the seam.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models

from config import Settings, get_settings
from src.confluence.extract import ExtractedPage
from src.ingest.chunk import Chunk, chunk_page
from src.ingest.parse import parse_blocks

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
_BM25_MODEL = "Qdrant/bm25"

# Stable namespace so a chunk_id always maps to the same Qdrant point id
# (re-index overwrites in place rather than duplicating).
_ID_NAMESPACE = uuid.UUID("6f8e2b0a-1c3d-4e5f-8a9b-0c1d2e3f4a5b")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _make_qdrant(settings: Settings) -> QdrantClient:
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def _payload(page: ExtractedPage, chunk: Chunk) -> dict:
    return {
        "page_id": page.page_id,
        "chunk_id": chunk.chunk_id,
        "space_key": page.space_key,
        "title": page.title,
        "breadcrumb": chunk.breadcrumb,
        "url": page.url,
        "labels": page.labels,
        "author": page.author,
        "last_modified": page.last_modified,
        "content_hash": page.content_hash,
        "text": chunk.text,
    }


class Indexer:
    """Owns the Qdrant client, the dense embedder, and the BM25 model."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self.qdrant = _make_qdrant(self.settings)
        self._embed = OpenAI(
            base_url=self.settings.embed_base_url,
            api_key=self.settings.embed_api_key,
        )
        self._bm25 = SparseTextEmbedding(model_name=_BM25_MODEL)

    # -- collection ------------------------------------------------------
    def ensure_collection(self) -> None:
        if self.qdrant.collection_exists(self.collection):
            return
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self.settings.embed_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        # Indexes for filtering + per-page deletes.
        for field in ("page_id", "space_key"):
            self.qdrant.create_payload_index(
                self.collection, field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def recreate_collection(self) -> None:
        """Drop and rebuild (used by scripts/reindex.py in M5)."""
        if self.qdrant.collection_exists(self.collection):
            self.qdrant.delete_collection(self.collection)
        self.ensure_collection()

    # -- embedding -------------------------------------------------------
    def _embed_dense(self, texts: List[str]) -> List[List[float]]:
        resp = self._embed.embeddings.create(
            model=self.settings.embed_model, input=texts
        )
        return [d.embedding for d in resp.data]

    def _embed_sparse(self, texts: List[str]) -> List[models.SparseVector]:
        out = []
        for emb in self._bm25.embed(texts):
            out.append(
                models.SparseVector(
                    indices=emb.indices.tolist(), values=emb.values.tolist()
                )
            )
        return out

    # -- indexing --------------------------------------------------------
    def index_page(self, page: ExtractedPage) -> int:
        """Parse -> chunk -> embed -> replace all points for this page.

        Deleting first makes updates and shrink-in-size correct (stale chunks
        never linger). Returns the number of chunks upserted.
        """
        blocks = parse_blocks(page.body_html)
        chunks = chunk_page(
            page,
            blocks,
            max_tokens=self.settings.chunk_max_tokens,
            hard_max_tokens=self.settings.chunk_hard_max_tokens,
        )
        self.delete_page(page.page_id)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        dense = self._embed_dense(texts)
        sparse = self._embed_sparse(texts)

        points = [
            models.PointStruct(
                id=_point_id(c.chunk_id),
                vector={DENSE_VECTOR: d, SPARSE_VECTOR: s},
                payload=_payload(page, c),
            )
            for c, d, s in zip(chunks, dense, sparse)
        ]
        self.qdrant.upsert(self.collection, points=points)
        return len(points)

    def delete_page(self, page_id: str) -> None:
        self.qdrant.delete(
            self.collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="page_id", match=models.MatchValue(value=page_id)
                    )
                ]
            ),
        )

    def count(self) -> int:
        return self.qdrant.count(self.collection, exact=True).count

    def close(self) -> None:
        self.qdrant.close()
