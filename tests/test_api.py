"""Offline API unit tests: webhook page-id extraction across payload shapes."""

from __future__ import annotations

from src.api.main import _extract_page_id


def test_extract_page_id_confluence_page_shape():
    assert _extract_page_id({"event": "page_updated", "page": {"id": 123}}) == "123"


def test_extract_page_id_content_shape():
    assert _extract_page_id({"content": {"id": "456"}}) == "456"


def test_extract_page_id_flat_shapes():
    assert _extract_page_id({"page_id": "789"}) == "789"
    assert _extract_page_id({"id": "42"}) == "42"


def test_extract_page_id_missing_returns_none():
    assert _extract_page_id({"event": "page_updated"}) is None
    assert _extract_page_id({}) is None
