"""Crawl the allowlisted Confluence spaces from scratch and report counts.

M1: extraction + safety rule + content-hash gate + page-state persistence. No
embeddings yet — the `on_index` callback is a no-op here, so a re-run re-pulls
only pages whose content_hash changed.

Usage:
    python -m scripts.full_sync
    python -m scripts.full_sync --since "2026-09-01 00:00"   # delta re-pull
"""

from __future__ import annotations

import argparse
import sys

from config import get_settings
from src.confluence.client import ConfluenceClient, ConfluenceSpaceNotFound
from src.confluence.sync import PageStateStore, delta_sync, full_sync


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
    args = parser.parse_args()

    settings = get_settings()
    store = PageStateStore(settings.page_state_db)

    try:
        with ConfluenceClient(
            settings.confluence_base_url, settings.confluence_pat
        ) as client:
            if args.since:
                _log(
                    f"Delta sync since {args.since!r} over spaces "
                    f"{settings.confluence_spaces}"
                )
                stats = delta_sync(
                    client=client,
                    store=store,
                    space_keys=settings.confluence_spaces,
                    since=args.since,
                    log=_log,
                )
            else:
                _log(f"Full sync over spaces {settings.confluence_spaces}")
                stats = full_sync(
                    client=client,
                    store=store,
                    space_keys=settings.confluence_spaces,
                    log=_log,
                )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
