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
from src.confluence.sync import (
    PageStateStore,
    SyncStats,
    full_sync,
    has_read_restrictions,
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

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

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
