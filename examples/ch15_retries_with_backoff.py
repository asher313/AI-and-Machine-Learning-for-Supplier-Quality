# Chapter 15 teaching listing. Supply the inputs described in the text.
import random
import time
import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import anthropic
import structlog
log = structlog.get_logger()

RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.ConflictError,
)


def _retryable(exc):
    return (
        isinstance(exc, RETRYABLE)
        or getattr(exc, "status_code", None) == 408
    )


def _delay(exc, attempt, base, cap):
    headers = getattr(
        getattr(exc, "response", None), "headers", {}
    )
    hint = None
    try:
        if headers.get("retry-after-ms"):
            hint = float(headers["retry-after-ms"]) / 1000
        elif headers.get("retry-after"):
            raw = headers["retry-after"]
            try:
                hint = float(raw)
            except ValueError:
                dt = parsedate_to_datetime(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                hint = (dt - datetime.now(UTC)).total_seconds()
    except (ValueError, TypeError, OverflowError):
        hint = None
    # Full jitter is capped; a valid server hint may require a longer wait.
    wait = random.uniform(
        0, min(cap, base * 2 ** min(attempt, 30))
    )
    return (
        max(wait, hint)
        if hint is not None and math.isfinite(hint) and hint >= 0
        else wait
    )


def _check_retry(attempts, base, cap):
    if (
        not isinstance(attempts, int)
        or attempts < 1
        or not all(
            math.isfinite(v) and v > 0 for v in (base, cap)
        )
    ):
        raise ValueError(
            "positive attempt count, base and cap required"
        )


def with_retry(
    fn, *args, attempts=5, base=1.0, cap=30.0, **kwargs
):
    """Retry requests only; never wrap non-idempotent tool side effects."""
    _check_retry(attempts, base, cap)
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except anthropic.APIError as exc:
            if not _retryable(exc) or attempt == attempts - 1:
                raise
            delay = _delay(exc, attempt, base, cap)
            log.warning(
                "llm_retry",
                attempt=attempt + 1,
                error=type(exc).__name__,
                sleep_s=round(delay, 1),
            )
            time.sleep(delay)
