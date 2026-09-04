"""Heading-aware chunking with breadcrumb context.

Walk the parsed blocks, tracking the current heading path (h1-h4). Accumulate
text under each heading into chunks that stay within a token budget; keep tables
and code blocks whole (they may exceed the budget — that is intentional). Every
chunk is prefixed with its breadcrumb: "Space > Parent > Page > Section > ...".

Token counting is approximate (chars/4 by default); the goal is stable, roughly
even chunk sizes, not exact model tokenization. The counter is injectable so
tests can use a deterministic word counter.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.confluence.extract import ExtractedPage
from src.ingest.parse import Block

DEFAULT_MAX_TOKENS = 512

# Hard ceiling that even "whole" tables/code may not exceed — the embedding
# endpoint rejects inputs over its context window (jina-v2: 8192 real tokens).
# Our estimate runs ~1.26x low vs. the real tokenizer on German/technical text,
# so 5000 estimated (~6300 real) keeps a safe margin. A block above this is
# split on natural boundaries (table rows / code lines).
DEFAULT_HARD_MAX_TOKENS = 5000

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Deterministic, model-agnostic."""
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Chunk:
    page_id: str
    ordinal: int
    chunk_id: str
    breadcrumb: str
    text: str  # breadcrumb-prefixed final text
    token_count: int
    kind: str  # "text" | "table" | "code"
    whole: bool  # intentionally-whole table/code (allowed to exceed max_tokens)


def _prefix(breadcrumb: str, body: str) -> str:
    return f"{breadcrumb}\n\n{body}"


def _split_text_to_budget(text: str, budget: int, count: TokenCounter) -> List[str]:
    """Greedily pack sentences into pieces each <= budget tokens.

    A single oversized sentence is hard-split on word boundaries.
    """
    if count(text) <= budget:
        return [text]

    pieces: List[str] = []
    current: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal current, cur_tokens
        if current:
            pieces.append(" ".join(current).strip())
            current = []
            cur_tokens = 0

    for sentence in _SENT_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        st = count(sentence)
        if st > budget:
            # Hard-split the long sentence by words.
            flush()
            words = sentence.split()
            buf: List[str] = []
            for w in words:
                buf.append(w)
                if count(" ".join(buf)) >= budget:
                    pieces.append(" ".join(buf))
                    buf = []
            if buf:
                current = buf
                cur_tokens = count(" ".join(buf))
            continue
        if cur_tokens + st > budget:
            flush()
        current.append(sentence)
        cur_tokens += st
    flush()
    return [p for p in pieces if p]


def _split_lines_to_budget(
    body: str, budget: int, count: TokenCounter, *, repeat_header: bool
) -> List[str]:
    """Split a multi-line block (table/code) into pieces <= budget tokens.

    Splits only on line boundaries so rows/lines stay intact. For tables,
    `repeat_header` prepends the first line to every piece for context.
    """
    lines = body.split("\n")
    if not lines:
        return []
    header = lines[0] if repeat_header else None
    pieces: List[str] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            body_lines = ([header] + current) if (header and current != []) else current
            pieces.append("\n".join(body_lines))
            current = []

    start = 1 if repeat_header else 0
    base = [header] if repeat_header else []
    for line in lines[start:]:
        trial = "\n".join(base + current + [line])
        if current and count(trial) > budget:
            flush()
        current.append(line)
    flush()
    # A single line longer than budget is emitted as-is (can't split further
    # without breaking the row); the hard cap has margin to absorb it.
    return [p for p in pieces if p.strip()]


def chunk_page(
    page: ExtractedPage,
    blocks: List[Block],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
    count_tokens: TokenCounter = estimate_tokens,
) -> List[Chunk]:
    """Split a page's blocks into breadcrumb-prefixed chunks."""
    chunks: List[Chunk] = []
    heading_stack: List[Tuple[int, str]] = []  # (level, text), levels strictly increasing
    text_buffer: List[str] = []

    def current_breadcrumb() -> str:
        parts = [page.breadcrumb] + [h[1] for h in heading_stack]
        return " > ".join(p for p in parts if p)

    def emit(body: str, kind: str, whole: bool) -> None:
        if not body.strip():
            return
        breadcrumb = current_breadcrumb()
        text = _prefix(breadcrumb, body)
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                page_id=page.page_id,
                ordinal=ordinal,
                chunk_id=f"{page.page_id}:{ordinal}",
                breadcrumb=breadcrumb,
                text=text,
                token_count=count_tokens(text),
                kind=kind,
                whole=whole,
            )
        )

    def flush_text() -> None:
        nonlocal text_buffer
        if not text_buffer:
            return
        body = "\n\n".join(text_buffer)
        breadcrumb = current_breadcrumb()
        # Reserve breadcrumb tokens so the full chunk stays within budget.
        reserve = count_tokens(_prefix(breadcrumb, ""))
        budget = max(1, max_tokens - reserve)
        for piece in _split_text_to_budget(body, budget, count_tokens):
            emit(piece, "text", whole=False)
        text_buffer = []

    def emit_whole_or_split(body: str, kind: str) -> None:
        """Emit a table/code block whole, or split it on line boundaries if it
        would blow the hard ceiling (embedder context limit)."""
        breadcrumb = current_breadcrumb()
        reserve = count_tokens(_prefix(breadcrumb, ""))
        hard_budget = max(1, hard_max_tokens - reserve)
        if count_tokens(body) <= hard_budget:
            emit(body, kind, whole=True)
            return
        pieces = _split_lines_to_budget(
            body, hard_budget, count_tokens, repeat_header=(kind == "table")
        )
        for piece in pieces:
            emit(piece, kind, whole=False)

    for block in blocks:
        if block.kind == "heading":
            flush_text()  # chunks never span a heading boundary
            level = block.level or 4
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.text))
        elif block.kind == "table":
            flush_text()
            emit_whole_or_split(block.text, "table")
        elif block.kind == "code":
            flush_text()
            emit_whole_or_split(block.text, "code")
        else:  # text
            text_buffer.append(block.text)

    flush_text()
    return chunks
