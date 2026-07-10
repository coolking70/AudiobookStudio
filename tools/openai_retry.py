"""Small retry policies shared by live evaluation scripts."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def call_with_404_backoff(
    operation: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry transient provider routing 404s with bounded exponential backoff."""
    for attempt in range(max(0, retries) + 1):
        try:
            return operation()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            if status != 404 or attempt >= retries:
                raise
            sleep(base_delay * (2 ** attempt))
    raise RuntimeError("unreachable")
