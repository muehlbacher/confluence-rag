"""Small retry helper with exponential backoff for flaky HTTP calls.

Retries transport errors (connect/read timeouts, resets) and retryable status
codes (429, 5xx). Does NOT retry other 4xx (e.g. 404, 401) — those are terminal.
No external dependency; the OpenAI client already has its own retry, so this is
used for the raw httpx paths (Confluence REST, the reranker).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

import httpx

T = TypeVar("T")


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return False


def call_with_retry(
    do: Callable[[], T],
    *,
    what: str,
    logger: logging.Logger,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
) -> T:
    """Call `do()`, retrying retryable failures with capped exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return do()
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            if attempt >= attempts or not is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, base_delay)  # jitter
            logger.warning(
                "retrying what=%s attempt=%d/%d after=%.2fs err=%s",
                what, attempt, attempts, delay, type(exc).__name__,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
