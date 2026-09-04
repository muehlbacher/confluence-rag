"""Turn a raw Confluence page (API dict) into a normalized `ExtractedPage`.

M1 only needs clean text + full metadata + a stable content hash. The rich
block/heading parsing for chunking lands in M2 (`ingest/parse.py`); here we do
just enough HTML-to-text to compute a hash that changes when the page's visible
content changes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from selectolax.parser import HTMLParser

_WS_RE = re.compile(r"\s+")

# Non-content nodes whose text is never page content. Confluence's rendered
# body.view carries inline <style>/<script> from macros (e.g. RefinedWiki emits
# CSS with a fresh per-render UUID in every rule), which both pollutes chunk
# text and makes the content hash unstable. Drop them before extracting text.
# NOTE: fuller macro-chrome stripping (nav, empty structural nodes) is M2's job
# in ingest/parse.py; here we do the minimum needed for a stable hash.
_DROP_TAGS = ("script", "style", "noscript", "template")


@dataclass
class ExtractedPage:
    page_id: str
    space_key: str
    title: str
    url: str
    labels: List[str] = field(default_factory=list)
    author: str = ""
    last_modified: str = ""
    ancestors: List[str] = field(default_factory=list)  # ordered root -> parent titles
    breadcrumb: str = ""  # "Space > Parent > Page"
    body_html: str = ""  # raw body.view HTML (kept for M2 parsing)
    text: str = ""  # normalized plain text
    content_hash: str = ""


def html_to_text(html: str) -> str:
    """Strip HTML to normalized visible text (whitespace-collapsed)."""
    if not html:
        return ""
    tree = HTMLParser(html)
    for node in tree.css(",".join(_DROP_TAGS)):
        node.decompose()
    text = tree.text(separator=" ")
    return _WS_RE.sub(" ", text).strip()


def compute_content_hash(title: str, text: str) -> str:
    """sha256 over normalized (title + body) text.

    Title is included so a title-only edit still triggers re-indexing.
    """
    h = hashlib.sha256()
    h.update(title.strip().encode("utf-8"))
    h.update(b"\n")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _build_url(raw: Dict[str, Any], base_url: str = "") -> str:
    # In a list response `_links.base` is only at the top level, not per result,
    # so per-page `_links` often carries just the relative `webui`. Fall back to
    # the known instance base_url to always produce an absolute, clickable URL.
    links = raw.get("_links", {}) or {}
    base = ((links.get("base") or base_url) or "").rstrip("/")
    webui = links.get("webui") or ""
    if base and webui:
        return f"{base}{webui}"
    return webui or base


def extract_page(raw: Dict[str, Any], base_url: str = "") -> ExtractedPage:
    """Build an `ExtractedPage` from a single content result dict.

    `base_url` is the Confluence instance base, used to absolutize the citation
    URL when the per-page `_links` only carry a relative path.
    """
    page_id = str(raw.get("id", ""))
    title = raw.get("title", "") or ""

    space_key = ((raw.get("space") or {}).get("key")) or ""

    version = raw.get("version") or {}
    author = ((version.get("by") or {}).get("displayName")) or ""
    last_modified = version.get("when") or ""

    labels = [
        lbl.get("name", "")
        for lbl in (((raw.get("metadata") or {}).get("labels") or {}).get("results") or [])
        if lbl.get("name")
    ]

    ancestors = [a.get("title", "") for a in (raw.get("ancestors") or []) if a.get("title")]

    body_html = (((raw.get("body") or {}).get("view") or {}).get("value")) or ""
    text = html_to_text(body_html)

    breadcrumb = " > ".join([p for p in ([space_key] + ancestors + [title]) if p])

    return ExtractedPage(
        page_id=page_id,
        space_key=space_key,
        title=title,
        url=_build_url(raw, base_url),
        labels=labels,
        author=author,
        last_modified=last_modified,
        ancestors=ancestors,
        breadcrumb=breadcrumb,
        body_html=body_html,
        text=text,
        content_hash=compute_content_hash(title, text),
    )
