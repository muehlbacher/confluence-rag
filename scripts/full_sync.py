"""Crawl the allowlisted Confluence spaces and index them into Qdrant.

Pipeline (M2): extract -> safety rule + content-hash gate -> parse -> chunk ->
embed (dense) + BM25 sparse -> upsert to Qdrant. A re-run re-pulls only pages
whose content_hash changed, so an unchanged corpus is a no-op.

Usage:
    python -m scripts.full_sync                      # incremental full crawl
    python -m scripts.full_sync --recreate           # drop collection + rebuild
    python -m scripts.full_sync --since "2026-09-01 00:00"   # delta re-pull
    python -m scripts.full_sync --no-index           # extraction only (M1 behaviour)
"""

from __future__ import annotations

import argparse
import sys

from config import get_settings
from src.confluence.client import ConfluenceClient, ConfluenceSpaceNotFound
from src.confluence.sync import PageStateStore, delta_sync, full_sync
from src.ingest.index import Indexer


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default=None,
        help='Delta mode: only pages modified at/after this timestamp, '
        'e.g. "2026-09-01 00:00". Omit for a full crawl.',
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the Qdrant collection and page-state first.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Extract + gate only; do not embed or write to Qdrant.",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = PageStateStore(settings.page_state_db)

    indexer = None
    on_index = None
    if not args.no_index:
        indexer = Indexer(settings)
        if args.recreate:
            _log("Recreating Qdrant collection and clearing page-state")
            indexer.recreate_collection()
            store.clear()
        else:
            indexer.ensure_collection()
        on_index = indexer.index_page

    try:
        with ConfluenceClient(
            settings.confluence_base_url, settings.confluence_pat
        ) as client:
            common = dict(
                client=client,
                store=store,
                space_keys=settings.confluence_spaces,
                log=_log,
            )
            if on_index is not None:
                common["on_index"] = on_index
            if args.since:
                _log(f"Delta sync since {args.since!r} over {settings.confluence_spaces}")
                stats = delta_sync(since=args.since, **common)
            else:
                _log(f"Full sync over {settings.confluence_spaces}")
                stats = full_sync(**common)
    except ConfluenceSpaceNotFound as exc:
        _log(f"ERROR: {exc}")
        return 2
    finally:
        store.close()

    s = stats.as_dict()
    print(
        "pages seen={seen} indexed={indexed} "
        "skipped-restricted={skipped_restricted} "
        "skipped-unchanged={skipped_unchanged} errors={errors}".format(**s)
    )
    if indexer is not None:
        print(f"qdrant points={indexer.count()}")
        indexer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
