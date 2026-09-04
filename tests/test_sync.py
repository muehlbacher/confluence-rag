"""M1 acceptance tests: safety rule (allowlist + restriction skip), content-hash
gate, and reported counts — driven through a mocked Confluence server so the
real client (pagination, restriction lookup, extraction) is exercised end to end.

(M5 will extend this file with webhook create/update/delete propagation.)
"""

from __future__ import annotations

import json
from typing import Dict, List, Set

import httpx
import pytest

from src.confluence.client import ConfluenceClient
from src.confluence.extract import compute_content_hash, html_to_text
from src.confluence.sync import (
    PageStateStore,
    SyncStats,
    delete_page,
    full_sync,
    has_read_restrictions,
    sync_page_by_id,
)


def make_page(page_id: str, space_key: str, title: str, body: str) -> Dict:
    """A raw content dict shaped like the API's expand=body.view,version,...."""
    return {
        "id": page_id,
        "type": "page",
        "status": "current",
        "title": title,
        "space": {"key": space_key},
        "version": {"when": "2026-09-01T10:00:00.000Z", "by": {"displayName": "Ada"}},
        "ancestors": [{"title": "Home"}],
        "metadata": {"labels": {"results": [{"name": "runbook"}]}},
        "body": {"view": {"value": f"<h1>{title}</h1><p>{body}</p>"}},
        "_links": {"base": "https://cf.example", "webui": f"/display/{space_key}/{page_id}"},
    }


class FakeConfluence:
    """In-memory Confluence backing an httpx.MockTransport.

    `pages` maps space_key -> list of raw page dicts. `restricted` is the set of
    page ids that carry a read restriction.
    """

    def __init__(self, pages: Dict[str, List[Dict]], restricted: Set[str]):
        self.pages = pages
        self.restricted = restricted

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _by_id(self, page_id: str):
        for items in self.pages.values():
            for p in items:
                if p["id"] == page_id:
                    return p
        return None

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        # Single page fetch: /rest/api/content/{id} (no trailing sub-resource).
        if path.startswith("/rest/api/content/") and path.count("/") == 4:
            page_id = path.split("/")[4]
            page = self._by_id(page_id)
            if page is None:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json=page)

        if path == "/rest/api/content":
            space = request.url.params.get("spaceKey")
            start = int(request.url.params.get("start", "0"))
            limit = int(request.url.params.get("limit", "50"))
            items = self.pages.get(space, [])
            window = items[start : start + limit]
            has_next = start + limit < len(items)
            body = {"results": window, "size": len(window)}
            if has_next:
                body["_links"] = {"next": f"/rest/api/content?start={start + limit}"}
            else:
                body["_links"] = {}
            return httpx.Response(200, json=body)

        if path.startswith("/rest/api/content/") and path.endswith(
            "/restriction/byOperation/read"
        ):
            page_id = path.split("/")[4]
            if page_id in self.restricted:
                payload = {
                    "operation": "read",
                    "restrictions": {
                        "user": {"results": [{"username": "bob"}], "size": 1},
                        "group": {"results": [], "size": 0},
                    },
                }
            else:
                payload = {
                    "operation": "read",
                    "restrictions": {
                        "user": {"results": [], "size": 0},
                        "group": {"results": [], "size": 0},
                    },
                }
            return httpx.Response(200, json=payload)

        return httpx.Response(404, json={"message": f"unhandled {path}"})


def make_client(fake: FakeConfluence) -> ConfluenceClient:
    http = httpx.Client(base_url="https://cf.example", transport=fake.transport())
    return ConfluenceClient("https://cf.example", "test-pat", client=http)


# --- unit: restriction interpretation ---------------------------------------

def test_has_read_restrictions_detects_user_and_group():
    assert has_read_restrictions(
        {"restrictions": {"user": {"size": 1, "results": [{}]}, "group": {"size": 0}}}
    )
    assert has_read_restrictions(
        {"restrictions": {"user": {"size": 0}, "group": {"size": 2, "results": [{}, {}]}}}
    )


def test_has_read_restrictions_false_when_empty():
    assert not has_read_restrictions(
        {"restrictions": {"user": {"size": 0, "results": []},
                          "group": {"size": 0, "results": []}}}
    )
    # top-level shape (no "restrictions" wrapper)
    assert not has_read_restrictions({"user": {"size": 0}, "group": {"size": 0}})


# --- unit: text extraction drops non-content nodes ---------------------------

def test_html_to_text_drops_style_and_script():
    # Rendered macros emit inline <style>/<script>; their text is not content.
    html = (
        "<style>.rwui_id_abc {color: #fff;}</style>"
        "<p>Real content here.</p>"
        "<script>var x = 1;</script>"
    )
    text = html_to_text(html)
    assert text == "Real content here."
    assert "rwui" not in text and "color" not in text


def test_content_hash_stable_against_per_render_css_uuids():
    # Same visible content, different per-render UUID in a <style> block — the
    # hash must not change (this is what broke the M1 no-op re-sync live).
    body_a = "<style>.rwui_id_11111111 {color:#fff}</style><p>Docs body.</p>"
    body_b = "<style>.rwui_id_99999999 {color:#fff}</style><p>Docs body.</p>"
    ha = compute_content_hash("Title", html_to_text(body_a))
    hb = compute_content_hash("Title", html_to_text(body_b))
    assert ha == hb


# --- acceptance: counts + restriction skip -----------------------------------

def test_full_sync_counts_and_skips_restricted(tmp_path):
    pages = {
        "ENG": [
            make_page("1", "ENG", "Alpha", "first"),
            make_page("2", "ENG", "Beta", "second"),
            make_page("3", "ENG", "Secret", "restricted body"),  # restricted
        ],
        "OPS": [make_page("4", "OPS", "Gamma", "ops doc")],
    }
    fake = FakeConfluence(pages, restricted={"3"})
    store = PageStateStore(str(tmp_path / "state.db"))

    logs: List[str] = []
    stats = full_sync(
        client=make_client(fake),
        store=store,
        space_keys=["ENG", "OPS"],
        log=logs.append,
    )

    assert stats.seen == 4
    assert stats.indexed == 3
    assert stats.skipped_restricted == 1
    assert stats.errors == 0
    # The restricted page is provably skipped and never enters page-state.
    assert store.get_hash("3") is None
    assert any("SKIP restricted page_id=3" in m for m in logs)
    assert set(store.all_page_ids()) == {"1", "2", "4"}


# --- acceptance: content-hash gate on re-run ---------------------------------

def test_rerun_repulls_only_changed_pages(tmp_path):
    pages = {"ENG": [make_page("1", "ENG", "Alpha", "v1"),
                     make_page("2", "ENG", "Beta", "stable")]}
    fake = FakeConfluence(pages, restricted=set())
    store = PageStateStore(str(tmp_path / "state.db"))
    client = make_client(fake)

    first = full_sync(client=client, store=store, space_keys=["ENG"])
    assert first.indexed == 2
    assert first.skipped_unchanged == 0

    # Edit page 1's body; page 2 unchanged.
    pages["ENG"][0] = make_page("1", "ENG", "Alpha", "v2-edited")
    fake.pages = pages

    logs: List[str] = []
    second = full_sync(
        client=make_client(fake), store=store, space_keys=["ENG"], log=logs.append
    )

    assert second.seen == 2
    assert second.indexed == 1  # only the edited page
    assert second.skipped_unchanged == 1
    assert any("INDEX page_id=1" in m for m in logs)
    assert not any("INDEX page_id=2" in m for m in logs)


class FakeIndexer:
    """Records index/delete operations, standing in for the real Qdrant indexer."""

    def __init__(self):
        self.indexed = []  # list of (page_id, content_hash)
        self.deleted = []

    def index_page(self, page):
        self.indexed.append((page.page_id, page.content_hash))

    def delete_page(self, page_id):
        self.deleted.append(page_id)


# --- M5: webhook create / update / delete propagation ------------------------

def test_webhook_create_indexes_new_page(tmp_path):
    pages = {"ENG": [make_page("10", "ENG", "New", "brand new page")]}
    fake = FakeConfluence(pages, restricted=set())
    store = PageStateStore(str(tmp_path / "s.db"))
    idx = FakeIndexer()

    reason = sync_page_by_id("10", client=make_client(fake), store=store, on_index=idx.index_page)

    assert reason is None  # indexed
    assert idx.indexed == [("10", store.get_hash("10"))]


def test_webhook_update_reindexes_changed_page(tmp_path):
    pages = {"ENG": [make_page("10", "ENG", "Doc", "v1")]}
    fake = FakeConfluence(pages, restricted=set())
    store = PageStateStore(str(tmp_path / "s.db"))
    idx = FakeIndexer()

    sync_page_by_id("10", client=make_client(fake), store=store, on_index=idx.index_page)
    first_hash = store.get_hash("10")

    # Page edited in Confluence.
    pages["ENG"][0] = make_page("10", "ENG", "Doc", "v2 edited content")
    fake.pages = pages
    reason = sync_page_by_id("10", client=make_client(fake), store=store, on_index=idx.index_page)

    assert reason is None
    assert len(idx.indexed) == 2
    assert store.get_hash("10") != first_hash  # hash advanced


def test_webhook_update_unchanged_is_noop(tmp_path):
    pages = {"ENG": [make_page("10", "ENG", "Doc", "same")]}
    fake = FakeConfluence(pages, restricted=set())
    store = PageStateStore(str(tmp_path / "s.db"))
    idx = FakeIndexer()

    sync_page_by_id("10", client=make_client(fake), store=store, on_index=idx.index_page)
    reason = sync_page_by_id("10", client=make_client(fake), store=store, on_index=idx.index_page)

    assert reason == "unchanged"
    assert len(idx.indexed) == 1  # not re-indexed


def test_webhook_update_restricted_page_is_skipped(tmp_path):
    pages = {"ENG": [make_page("10", "ENG", "Secret", "now restricted")]}
    fake = FakeConfluence(pages, restricted={"10"})
    store = PageStateStore(str(tmp_path / "s.db"))
    idx = FakeIndexer()

    reason = sync_page_by_id("10", client=make_client(fake), store=store, on_index=idx.index_page)

    assert reason == "restricted"
    assert idx.indexed == []
    assert store.get_hash("10") is None


def test_webhook_delete_removes_page(tmp_path):
    store = PageStateStore(str(tmp_path / "s.db"))
    idx = FakeIndexer()
    # seed page-state as if it were indexed
    pages = {"ENG": [make_page("10", "ENG", "Doc", "content")]}
    sync_page_by_id("10", client=make_client(FakeConfluence(pages, set())),
                    store=store, on_index=idx.index_page)
    assert store.get_hash("10") is not None

    delete_page("10", store=store, on_delete=idx.delete_page)

    assert idx.deleted == ["10"]
    assert store.get_hash("10") is None


def test_index_callback_receives_extracted_page(tmp_path):
    pages = {"ENG": [make_page("1", "ENG", "Alpha", "hello world")]}
    fake = FakeConfluence(pages, restricted=set())
    store = PageStateStore(str(tmp_path / "state.db"))

    captured = []
    full_sync(
        client=make_client(fake),
        store=store,
        space_keys=["ENG"],
        on_index=captured.append,
    )

    assert len(captured) == 1
    page = captured[0]
    assert page.title == "Alpha"
    assert page.space_key == "ENG"
    assert page.author == "Ada"
    assert page.breadcrumb == "ENG > Home > Alpha"
    assert page.url == "https://cf.example/display/ENG/1"
    assert page.labels == ["runbook"]
    assert "hello world" in page.text
    assert len(page.content_hash) == 64
