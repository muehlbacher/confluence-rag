"""Confluence Data Center REST client.

Thin wrapper over httpx with PAT auth, pagination, CQL delta search, and the
read-restriction lookup that the v1 safety rule depends on. No parsing or
business logic lives here — callers get raw JSON dicts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import httpx

# Expansions we always want on a page: rendered body, version (author +
# lastModified), ancestors (for breadcrumbs), space (for the key), labels.
PAGE_EXPAND = "body.view,version,ancestors,space,metadata.labels"

_DEFAULT_PAGE_LIMIT = 50


class ConfluenceClient:
    """Authenticated client for a single Confluence DC instance."""

    def __init__(
        self,
        base_url: str,
        pat: str,
        *,
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # `client` injection keeps this testable (transport mocking) and lets
        # M5 supply a client configured with retry/backoff transport.
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {pat}"},
            timeout=timeout,
        )

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ConfluenceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low-level -------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    # -- pages -----------------------------------------------------------
    def iter_space_pages(
        self,
        space_key: str,
        *,
        expand: str = PAGE_EXPAND,
        limit: int = _DEFAULT_PAGE_LIMIT,
    ) -> Iterator[Dict[str, Any]]:
        """Yield every current page in a space, following `start` pagination.

        Stops when the response has no `_links.next`.
        """
        start = 0
        while True:
            data = self._get(
                "/rest/api/content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "status": "current",
                    "expand": expand,
                    "limit": limit,
                    "start": start,
                },
            )
            results = data.get("results", [])
            for page in results:
                yield page

            # Prefer the server's own pagination signal; fall back to size math.
            if not data.get("_links", {}).get("next"):
                break
            if not results:
                break
            start += len(results)

    def iter_changed_pages(
        self,
        since: str,
        space_keys: List[str],
        *,
        expand: str = PAGE_EXPAND,
        limit: int = _DEFAULT_PAGE_LIMIT,
    ) -> Iterator[Dict[str, Any]]:
        """Yield pages modified at/after `since` within the allowlisted spaces.

        `since` is a Confluence-friendly timestamp, e.g. "2026-09-01 13:45".
        """
        spaces = ", ".join(f'"{k}"' for k in space_keys)
        cql = (
            f'type = page and space in ({spaces}) '
            f'and lastModified >= "{since}" order by lastModified'
        )
        start = 0
        while True:
            data = self._get(
                "/rest/api/content/search",
                params={"cql": cql, "expand": expand, "limit": limit, "start": start},
            )
            results = data.get("results", [])
            for page in results:
                yield page
            if not data.get("_links", {}).get("next"):
                break
            if not results:
                break
            start += len(results)

    def get_page(self, page_id: str, *, expand: str = PAGE_EXPAND) -> Dict[str, Any]:
        """Fetch a single page by id (used by the webhook sync path in M5)."""
        return self._get(f"/rest/api/content/{page_id}", params={"expand": expand})

    # -- restrictions ----------------------------------------------------
    def get_read_restrictions(self, page_id: str) -> Dict[str, Any]:
        """Raw read-operation restrictions for a page.

        See `sync.has_read_restrictions` for interpretation.
        """
        return self._get(f"/rest/api/content/{page_id}/restriction/byOperation/read")
