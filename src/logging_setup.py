"""Central logging configuration.

Structured key=value log lines under the `confluence_rag.*` namespace. Call
`configure_logging()` once at process start (scripts, API startup).
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_ROOT = "confluence_rag"


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    root.handlers[:] = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT}.{name}")
