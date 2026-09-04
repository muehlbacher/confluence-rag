"""Reranker logic (mocked endpoint): threshold filtering, top_n cap, ordering.

Covers the M3 criterion: when no candidate clears RERANK_SCORE_MIN the reranker
returns empty (the M4 no-context signal).
"""

from __future__ import annotations

import json

import httpx
import pytest

from config import Settings
from src.retrieval.rerank import Reranker
from src.retrieval.search import Candidate


def make_candidates(n: int):
    return [
        Candidate(
            chunk_id=f"p{i}:0",
            page_id=f"p{i}",
            score=1.0,
            payload={"text": f"doc {i}", "page_id": f"p{i}", "chunk_id": f"p{i}:0"},
        )
        for i in range(n)
    ]


def reranker_with_scores(scores):
    """Reranker whose endpoint returns `scores[index]` for each document."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        results = [
            {"index": i, "relevance_score": scores[i]}
            for i in range(len(body["documents"]))
        ]
        return httpx.Response(200, json={"results": results})

    settings = Settings(
        confluence_base_url="https://x",
        confluence_pat="x",
        confluence_spaces="A",
        rerank_url="https://rerank.test/v1/rerank",
        rerank_top_n=5,
        rerank_score_min=0.3,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Reranker(settings, client=client)


def test_returns_empty_when_nothing_clears_threshold():
    r = reranker_with_scores([0.01, 0.05, 0.2, 0.29])
    out = r.rerank("q", make_candidates(4))
    assert out == []


def test_filters_below_threshold_and_orders_desc():
    # scores by candidate index; only 0.9, 0.5, 0.4 clear 0.3
    r = reranker_with_scores([0.4, 0.05, 0.9, 0.5])
    out = r.rerank("q", make_candidates(4))
    assert [x.page_id for x in out] == ["p2", "p3", "p0"]
    assert [round(x.score, 2) for x in out] == [0.9, 0.5, 0.4]


def test_top_n_cap():
    r = reranker_with_scores([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    out = r.rerank("q", make_candidates(6), top_n=3)
    assert len(out) == 3
    assert [x.page_id for x in out] == ["p0", "p1", "p2"]


def test_empty_candidates_short_circuits():
    r = reranker_with_scores([])
    assert r.rerank("q", []) == []


def test_score_min_override_zero_keeps_all():
    r = reranker_with_scores([0.4, 0.05, 0.9, 0.29])
    out = r.rerank("q", make_candidates(4), top_n=10, score_min=0.0)
    assert len(out) == 4  # nothing filtered
