# src/sqm_ai/timing.py
import time
from collections.abc import Iterator
from contextlib import contextmanager

import structlog

log = structlog.get_logger()


@contextmanager
def timed(event: str, **fields) -> Iterator[None]:
    """Log `event` with latency_ms when the block exits."""
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = round((time.perf_counter() - start) * 1000)
        log.info(event, latency_ms=ms, **fields)
