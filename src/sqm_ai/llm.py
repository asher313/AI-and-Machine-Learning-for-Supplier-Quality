# Chapter 15 — Asher at Northlake
# TODO(book): condensed in Chapter 15 — complete before production use.
# src/sqm_ai/llm.py
"""One place for every language-model call in sqm-ai."""
from __future__ import annotations

from anthropic import Anthropic, AsyncAnthropic

import structlog

log = structlog.get_logger()

# Snapshot: model ids as of September 2026. Change here only.
MODELS = {
    "frontier": "claude-opus-5",
    "standard": "claude-sonnet-5",
    "fast": "claude-haiku-4-5",
}

# USD per million tokens, list price, September 2026:
# (input, output, cache write, cache read)
PRICES = {
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}

# SDK absorbs one blip; with_retry() owns the long horizon.
client = Anthropic(max_retries=1, timeout=60.0)
aclient = AsyncAnthropic(max_retries=1, timeout=60.0)


def estimate_cost(model: str, usage) -> float:
    """Dollars for one response, from its usage block."""
    p_in, p_out, p_cw, p_cr = PRICES[model]
    written = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    return (
        usage.input_tokens * p_in
        + usage.output_tokens * p_out
        + written * p_cw
        + read * p_cr
    ) / 1_000_000


def log_usage(response, **fields) -> float:
    """Log tokens and cost for one response; never the text."""
    cost = estimate_cost(response.model, response.usage)
    log.info(
        "llm_call",
        model=response.model,
        request_id=response._request_id,
        stop=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read=response.usage.cache_read_input_tokens,
        cost_usd=round(cost, 6),
        **fields,
    )
    return cost

# with_retry() from 15.4 and pack_context() from 15.7 live
# here too (condensed: both shown earlier in this chapter).


# Chapter 15 — 15.4 Retries With Backoff (continued)
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


# Chapter 15 — 15.7 Counting Tokens and Fitting the Window (continued)
from sqm_ai.llm import MODELS, client


def count_tokens(
    text: str, model: str = MODELS["standard"],
) -> int:
    resp = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


# Chapter 15 — 15.7 Counting Tokens and Fitting the Window (continued)
def pack_context(
    chunks: list[tuple[str, int]],   # (text, token_count), ranked
    budget: int,
) -> list[str]:
    """Take chunks in rank order until the budget is spent."""
    used, kept = 0, []
    for text, n in chunks:
        if used + n > budget:
            break
        kept.append(text)
        used += n
    return kept


def context_budget(
    system: str, question: str,
    window: int = 200_000, reserve: int = 4_000,
) -> int:
    """Tokens left for context after prompt and answer."""
    fixed = count_tokens(system) + count_tokens(question)
    return window - fixed - reserve
