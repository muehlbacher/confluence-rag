"""Generation logic offline (stubbed search/rerank/LLM):

- no-context short-circuit when rerank is empty (model never called)
- model-side refusal maps to the no-context response
- citations come only from retrieved payloads and resolve; [n] filtering + dedupe
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation.answer import (
    NO_CONTEXT_ANSWER,
    Rag,
    _citations_from_answer,
)
from src.retrieval.rerank import Reranked


def make_reranked(page_id, ordinal=0, score=0.9, title=None, url=None):
    return Reranked(
        chunk_id=f"{page_id}:{ordinal}",
        page_id=page_id,
        score=score,
        payload={
            "page_id": page_id,
            "chunk_id": f"{page_id}:{ordinal}",
            "title": title or f"Title {page_id}",
            "url": url or f"https://cf/{page_id}",
            "last_modified": "2026-01-01T00:00:00.000Z",
            "text": f"text for {page_id}",
        },
    )


class StubSearcher:
    def search(self, question, **kw):
        return ["c"]  # opaque; the stub reranker ignores it


class StubReranker:
    def __init__(self, result):
        self.result = result

    def rerank(self, question, candidates, **kw):
        return self.result


class StubLLM:
    """Minimal openai-like client returning a canned message content."""

    def __init__(self, content):
        self.calls = 0
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        self._resp = SimpleNamespace(choices=[choice])

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                return outer._resp

        self.chat = SimpleNamespace(completions=_Completions())


def make_rag(reranked, llm_content):
    llm = StubLLM(llm_content)
    rag = Rag(
        searcher=StubSearcher(),
        reranker=StubReranker(reranked),
        llm=llm,
    )
    return rag, llm


# --- pure citation parsing ---------------------------------------------------

def test_citations_referenced_markers_only_and_dedup():
    chunks = [make_reranked("p1"), make_reranked("p1", ordinal=1), make_reranked("p2")]
    cites = _citations_from_answer("See [1] and [3].", chunks)
    assert [c.page_id for c in cites] == ["p1", "p2"]  # [1]->p1, [3]->p2, deduped


def test_citations_fallback_to_all_when_no_markers():
    chunks = [make_reranked("p1"), make_reranked("p2")]
    cites = _citations_from_answer("Antwort ohne Marker.", chunks)
    assert [c.page_id for c in cites] == ["p1", "p2"]


def test_citations_ignore_out_of_range_markers():
    chunks = [make_reranked("p1")]
    cites = _citations_from_answer("Bad ref [9].", chunks)
    assert cites == []  # [9] out of range -> no valid referenced source


# --- pipeline gates ----------------------------------------------------------

def test_no_context_when_rerank_empty_does_not_call_llm():
    rag, llm = make_rag([], "should not be used")
    res = rag.answer("frage")
    assert res.used_context is False
    assert res.answer == NO_CONTEXT_ANSWER
    assert res.citations == []
    assert llm.calls == 0  # model never invoked


def test_model_refusal_maps_to_no_context():
    rag, llm = make_rag([make_reranked("p1")], NO_CONTEXT_ANSWER)
    res = rag.answer("frage")
    assert res.used_context is False
    assert res.citations == []
    assert llm.calls == 1


def test_grounded_answer_has_resolving_citations():
    chunks = [make_reranked("p1", title="Alpha"), make_reranked("p2", title="Beta")]
    rag, _ = make_rag(chunks, "Die Antwort steht hier [1].")
    res = rag.answer("frage")
    assert res.used_context is True
    assert [c.page_id for c in res.citations] == ["p1"]
    assert res.citations[0].title == "Alpha"
    assert res.citations[0].url == "https://cf/p1"


def test_empty_model_content_maps_to_no_context():
    rag, _ = make_rag([make_reranked("p1")], None)
    res = rag.answer("frage")
    assert res.used_context is False
