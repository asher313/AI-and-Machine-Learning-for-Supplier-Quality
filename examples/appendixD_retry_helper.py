# Appendix D — D.6 The Retry Helper (Chapter 15.4)
import random
import time

import anthropic

import structlog

log = structlog.get_logger()

RETRYABLE = (
    anthropic.RateLimitError,        # 429
    anthropic.InternalServerError,   # 500 and 529
    anthropic.APIConnectionError,    # network, incl. timeouts
)


def with_retry(fn, *args, attempts: int = 5, base: float = 1.0,
               cap: float = 30.0, **kwargs):
    """Call fn(*args, **kwargs); back off on transient errors.

    Anything not in RETRYABLE is raised immediately: a 400 will
    not become a 200 by waiting.
    """
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except RETRYABLE as exc:
            if attempt == attempts - 1:
                raise
            delay = min(cap, base * 2 ** attempt)
            delay += random.uniform(0, delay / 2)      # jitter
            resp = getattr(exc, "response", None)
            hint = (
                resp.headers.get("retry-after") if resp else None
            )
            if hint:
                delay = max(delay, float(hint))
            log.warning(
                "llm_retry", attempt=attempt + 1,
                error=type(exc).__name__, sleep_s=round(delay, 1),
            )
            time.sleep(delay)
