"""Drop the Qdrant collection + page-state and rebuild from scratch.

A full, destructive rebuild — use after changing the embedding model/dimension,
the chunking, or when the index is suspected inconsistent. For incremental
updates use scripts/full_sync.py instead.

    python -m scripts.reindex
"""

from __future__ import annotations

from config import get_settings
from src.confluence.client import ConfluenceClient, ConfluenceSpaceNotFound
from src.confluence.sync import PageStateStore, full_sync
from src.ingest.index import Indexer
from src.logging_setup import configure_logging, get_logger

configure_logging()
_logger = get_logger("scripts.reindex")


def main() -> int:
    settings = get_settings()
    store = PageStateStore(settings.page_state_db)
    indexer = Indexer(settings)

    _logger.info("recreate collection=%s + clear page-state", settings.qdrant_collection)
    indexer.recreate_collection()
    store.clear()

    try:
        with ConfluenceClient(
            settings.confluence_base_url, settings.confluence_pat
        ) as client:
            stats = full_sync(
                client=client,
                store=store,
                space_keys=settings.confluence_spaces,
                on_index=indexer.index_page,
                log=_logger.info,
            )
    except ConfluenceSpaceNotFound as exc:
        _logger.error("%s", exc)
        return 2
    finally:
        store.close()

    s = stats.as_dict()
    print(
        "reindex complete: seen={seen} indexed={indexed} "
        "skipped-restricted={skipped_restricted} errors={errors}".format(**s)
    )
    print(f"qdrant points={indexer.count()}")
    indexer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
