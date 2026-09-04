"""Parse Confluence `body.view` HTML into clean, ordered content blocks.

The rendered view is a soup of nested layout <div>s with macro chrome. We walk
it in document order and emit a flat list of typed blocks:

- heading (with level 1-4; h5/h6 clamp to 4)
- text     (paragraphs, blockquotes, lists — flattened to text)
- table    (kept whole, rendered row-by-row)
- code     (kept whole)

Tables and code are emitted as single blocks so chunking can keep them intact.
Empty/whitespace-only nodes and non-content nodes (script/style/nav) are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from selectolax.parser import HTMLParser, Node

_WS_RE = re.compile(r"\s+")

# Non-content nodes: script/style (macro CSS/JS with per-render UUIDs — see
# extract.html_to_text) plus obvious navigation/breadcrumb chrome.
_DROP_TAGS = ("script", "style", "noscript", "template", "nav")

# Leaf content elements — we emit a block for these and do not descend further.
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT_TAGS = {"p", "blockquote"}
_LIST_TAGS = {"ul", "ol"}
_TABLE_TAG = "table"
_CODE_TAGS = {"pre"}

BlockKind = str  # "heading" | "text" | "table" | "code"


@dataclass
class Block:
    kind: BlockKind
    text: str
    level: Optional[int] = None  # only for headings (1-4)


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _render_table(node: Node) -> str:
    """Flatten a table to text, one row per line, cells separated by ' | '."""
    rows: List[str] = []
    for tr in node.css("tr"):
        cells = [
            _norm(cell.text(separator=" "))
            for cell in tr.iter()
            if cell.tag in ("td", "th")
        ]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _render_list(node: Node) -> str:
    """Flatten a list to '- item' lines (nested items collapse into their parent)."""
    lines: List[str] = []
    for li in node.iter():
        if li.tag == "li":
            text = _norm(li.text(separator=" "))
            if text:
                lines.append(f"- {text}")
    return "\n".join(lines)


def _walk(node: Node, blocks: List[Block]) -> None:
    for child in node.iter(include_text=False):
        tag = child.tag
        if tag in _DROP_TAGS:
            continue
        if tag in _HEADINGS:
            text = _norm(child.text(separator=" "))
            if text:
                level = min(int(tag[1]), 4)
                blocks.append(Block("heading", text, level=level))
        elif tag == _TABLE_TAG:
            text = _render_table(child)
            if text:
                blocks.append(Block("table", text))
        elif tag in _CODE_TAGS:
            text = child.text(separator="\n").strip("\n")
            if text.strip():
                blocks.append(Block("code", text))
        elif tag in _LIST_TAGS:
            text = _render_list(child)
            if text:
                blocks.append(Block("text", text))
        elif tag in _TEXT_TAGS:
            text = _norm(child.text(separator=" "))
            if text:
                blocks.append(Block("text", text))
        else:
            # Layout container (div/section/main/td/…): descend.
            _walk(child, blocks)


def parse_blocks(html: str) -> List[Block]:
    """Parse rendered body HTML into an ordered list of content blocks."""
    if not html:
        return []
    tree = HTMLParser(html)
    for node in tree.css(",".join(_DROP_TAGS)):
        node.decompose()
    root = tree.body or tree.root
    if root is None:
        return []
    blocks: List[Block] = []
    _walk(root, blocks)
    return blocks
