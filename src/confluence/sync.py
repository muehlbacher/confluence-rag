"""Sync orchestration: allowlist + restriction safety rule, content-hash gate,
and a small SQLite page-state store.

M1 stops at "would index" — the `on_index` callback is where M2 plugs in
embedding + Qdrant upsert. Here we only decide skip vs. (re)index and keep the
page-state store current so deltas are cheap.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .client import ConfluenceClient
from .extract import ExtractedPage, extract_page

# Callback invoked when a page needs (re)indexing. M2 supplies the real one
# (embed + upsert); M1 default is a no-op.
IndexFn = Callable[[ExtractedPage], None]

# Reason a page was skipped, for reporting.
SKIP_RESTRICTED = "restricted"
SKIP_UNCHANGED = "unchanged"


def has_read_restrictions(payload: Dict[str, Any]) -> bool:
    """True if the read-restriction payload names any user or group.

    Handles both shapes seen across DC versions: the object may carry the
    user/group sets under a `restrictions` key or at the top level.
    """
    block = payload.get("restrictions", payload) or {}
    for kind in ("user", "group"):
        section = block.get(kind) or {}
        size = section.get("size")
        results = section.get("results") or []
        if (size or 0) > 0 or len(results) > 0:
            return True
    return False


@dataclass
class SyncStats:
    seen: int = 0
    indexed: int = 0
    skipped_restricted: int = 0
    skipped_unchanged: int = 0
    deleted: int = 0
    errors: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "seen": self.seen,
            "indexed": self.indexed,
            "skipped_restricted": self.skipped_restricted,
            "skipped_unchanged": self.skipped_unchanged,
            "deleted": self.deleted,
            "errors": self.errors,
        }


class PageStateStore:
    """SQLite mapping page_id -> content_hash + last_modified + light metadata.

    Lets sync decide skip vs. re-embed without reading Qdrant. Kept deliberately
    small; the source of truth for chunks is Qdrant (from M2 on).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # check_same_thread off so the FastAPI webhook path (M5) can reuse it.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS page_state (
                    page_id       TEXT PRIMARY KEY,
                    content_hash  TEXT NOT NULL,
                    last_modified TEXT,
                    space_key     TEXT,
                    title         TEXT,
                    url           TEXT
                )
                """
            )

    def get_hash(self, page_id: str) -> Optional[str]:
        with closing(self._conn.execute(
            "SELECT content_hash FROM page_state WHERE page_id = ?", (page_id,)
        )) as cur:
            row = cur.fetchone()
        return row["content_hash"] if row else None

    def upsert(self, page: ExtractedPage) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO page_state
                    (page_id, content_hash, last_modified, space_key, title, url)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(page_id) DO UPDATE SET
                    content_hash  = excluded.content_hash,
                    last_modified = excluded.last_modified,
                    space_key     = excluded.space_key,
                    title         = excluded.title,
                    url           = excluded.url
                """,
                (
                    page.page_id,
                    page.content_hash,
                    page.last_modified,
                    page.space_key,
                    page.title,
                    page.url,
                ),
            )

    def all_page_ids(self, space_keys: Optional[Iterable[str]] = None) -> List[str]:
        if space_keys is None:
            sql, args = "SELECT page_id FROM page_state", ()
        else:
            keys = list(space_keys)
            placeholders = ",".join("?" * len(keys))
            sql = f"SELECT page_id FROM page_state WHERE space_key IN ({placeholders})"
            args = tuple(keys)
        with closing(self._conn.execute(sql, args)) as cur:
            return [r["page_id"] for r in cur.fetchall()]

    def delete(self, page_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM page_state WHERE page_id = ?", (page_id,))

    def clear(self) -> None:
        """Wipe all page-state (used with a collection recreate for a clean rebuild)."""
        with self._conn:
            self._conn.execute("DELETE FROM page_state")

    def close(self) -> None:
        self._conn.close()


def _noop_index(_page: ExtractedPage) -> None:
    """Default index callback for M1 (no embeddings yet)."""


def sync_page(
    raw: Dict[str, Any],
    *,
    client: ConfluenceClient,
    store: PageStateStore,
    stats: SyncStats,
    on_index: IndexFn = _noop_index,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Apply the safety rule + hash gate to one raw page.

    Returns None if indexed, or a SKIP_* reason string otherwise.
    """
    stats.seen += 1
    page = extract_page(raw, base_url=client.base_url)
    _log = log or (lambda _m: None)

    # v1 safety rule, part 2: skip any page with view restrictions.
    if has_read_restrictions(client.get_read_restrictions(page.page_id)):
        stats.skipped_restricted += 1
        _log(f"SKIP restricted page_id={page.page_id} title={page.title!r}")
        return SKIP_RESTRICTED

    # Content-hash gate: only (re)index when the visible content changed.
    if store.get_hash(page.page_id) == page.content_hash:
        stats.skipped_unchanged += 1
        return SKIP_UNCHANGED

    on_index(page)
    store.upsert(page)
    stats.indexed += 1
    _log(f"INDEX page_id={page.page_id} title={page.title!r} hash={page.content_hash[:12]}")
    return None


def full_sync(
    *,
    client: ConfluenceClient,
    store: PageStateStore,
    space_keys: List[str],
    on_index: IndexFn = _noop_index,
    log: Optional[Callable[[str], None]] = None,
) -> SyncStats:
    """Crawl every allowlisted space (safety rule part 1) and sync each page."""
    stats = SyncStats()
    for space_key in space_keys:
        for raw in client.iter_space_pages(space_key):
            try:
                sync_page(
                    raw, client=client, store=store, stats=stats,
                    on_index=on_index, log=log,
                )
            except Exception as exc:  # noqa: BLE001 - keep crawling other pages
                stats.errors += 1
                if log:
                    log(f"ERROR page_id={raw.get('id')} {type(exc).__name__}: {exc}")
    return stats


def delta_sync(
    *,
    client: ConfluenceClient,
    store: PageStateStore,
    space_keys: List[str],
    since: str,
    on_index: IndexFn = _noop_index,
    log: Optional[Callable[[str], None]] = None,
) -> SyncStats:
    """Re-pull only pages modified since `since` (CQL), then apply the hash gate."""
    stats = SyncStats()
    for raw in client.iter_changed_pages(since, space_keys):
        try:
            sync_page(
                raw, client=client, store=store, stats=stats,
                on_index=on_index, log=log,
            )
        except Exception as exc:  # noqa: BLE001
            stats.errors += 1
            if log:
                log(f"ERROR page_id={raw.get('id')} {type(exc).__name__}: {exc}")
    return stats
