# src/sqm_ai/llm.py
"""Anthropic adapter: lazy clients, bounded retries, token estimates and costs."""

from __future__ import annotations

import asyncio
import math
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache

import anthropic
import structlog

from sqm_ai.settings import get_settings

log = structlog.get_logger()
MODELS = {
    "frontier": "claude-opus-5",
    "standard": "claude-sonnet-5",
    "fast": "claude-haiku-4-5-20251001",
}
# Direct API standard text-token rates, checked September 11, 2026.
# Input, output, 5-minute cache write, cache read; 1h writes are 2x input.
PRICES = {
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-sonnet-5": (2.0, 10.0, 2.50, 0.20),
    "claude-haiku-4-5-20251001": (1.0, 5.0, 1.25, 0.10),
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.10),
}


def request_options(model, effort="low"):
    """Only send effort where supported; schema validity is not determinism."""
    if model in {MODELS["standard"], MODELS["frontier"]}:
        return {"output_config": {"effort": effort}}
    if model in {MODELS["fast"], "claude-haiku-4-5"}:
        return {}
    raise ValueError(
        "register the model's capabilities before calling it"
    )


@lru_cache
def get_client(asynchronous=False):
    key = get_settings().anthropic_api_key.get_secret_value()
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY is missing; configure the environment or .env"
        )
    cls = (
        anthropic.AsyncAnthropic
        if asynchronous
        else anthropic.Anthropic
    )
    return cls(api_key=key, max_retries=0, timeout=60.0)


class _LazyClient:
    def __init__(self, asynchronous=False):
        self.asynchronous = asynchronous

    def __getattr__(self, name):
        return getattr(get_client(self.asynchronous), name)


client = _LazyClient()
aclient = _LazyClient(True)


def text_response(response):
    if response.stop_reason != "end_turn":
        raise ValueError(
            f"incomplete or non-text response: {response.stop_reason}"
        )
    text = "".join(
        b.text for b in response.content if b.type == "text"
    )
    if not text:
        raise ValueError("response has no text")
    return text


def parsed_response(response):
    if (
        response.stop_reason != "end_turn"
        or response.parsed_output is None
    ):
        raise ValueError(
            "no complete validated structured response"
        )
    return response.parsed_output


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


async def awith_retry(
    fn, *args, attempts=5, base=1.0, cap=30.0, **kwargs
):
    _check_retry(attempts, base, cap)
    for attempt in range(attempts):
        try:
            return await fn(*args, **kwargs)
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
            await asyncio.sleep(delay)


def estimate_cost(model, usage):
    """Text-token estimate, not an invoice; excludes tools/residency modifiers."""
    if model not in PRICES:
        raise ValueError("no verified price for response model")
    p_in, p_out, p_5m, p_read = PRICES[model]
    written = usage.cache_creation_input_tokens or 0
    creation = getattr(usage, "cache_creation", None)
    one_hour = (
        getattr(creation, "ephemeral_1h_input_tokens", 0) or 0
    )
    five_min = (
        getattr(creation, "ephemeral_5m_input_tokens", written)
        if creation is not None
        else written
    )
    if five_min + one_hour != written:
        raise ValueError(
            "cache write breakdown does not match total"
        )
    tier = getattr(usage, "service_tier", None)
    if tier not in (None, "standard", "batch"):
        raise ValueError(
            "pricing modifier required for this service tier"
        )
    multiplier = 0.5 if tier == "batch" else 1.0
    return (
        multiplier
        * (
            usage.input_tokens * p_in
            + usage.output_tokens * p_out
            + five_min * p_5m
            + one_hour * 2 * p_in
            + (usage.cache_read_input_tokens or 0) * p_read
        )
        / 1_000_000
    )


def log_usage(response, **fields):
    """Log selected operational metadata, never arbitrary caller payloads."""
    allowed = {
        "agent",
        "step",
        "tool",
        "stage",
        "crew",
        "role",
        "trace_id",
        "build",
    }
    if set(fields) - allowed:
        raise ValueError("unapproved usage-log fields")
    cost = estimate_cost(response.model, response.usage)
    log.info(
        "llm_call",
        model=response.model,
        request_id=getattr(response, "_request_id", None),
        stop=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_written=response.usage.cache_creation_input_tokens,
        cache_read=response.usage.cache_read_input_tokens,
        cost_usd=round(cost, 6),
        **fields,
    )
    return cost


def count_tokens(
    text=None,
    model=MODELS["standard"],
    *,
    messages=None,
    system=None,
    tools=None,
):
    """Provider estimate for the complete request shape, before output reserve."""
    if messages is None:
        if text is None:
            raise ValueError("text or messages required")
        messages = [{"role": "user", "content": text}]
    request = {"model": model, "messages": messages}
    if system is not None:
        request["system"] = system
    if tools is not None:
        request["tools"] = tools
    return with_retry(
        client.messages.count_tokens, **request
    ).input_tokens


def pack_context(chunks, budget):
    """Keep a relevance-ordered prefix; lengths must use the target tokenizer."""
    if budget < 0:
        raise ValueError("nonnegative context budget required")
    used, kept = 0, []
    for text, n in chunks:
        if not isinstance(n, int) or n < 0:
            raise ValueError(
                "token counts must be nonnegative integers"
            )
        if used + n > budget:
            break
        kept.append(text)
        used += n
    return kept


def context_budget(
    system,
    question,
    window=200_000,
    reserve=4_000,
    *,
    model=MODELS["standard"],
    tools=None,
    margin=512,
):
    if window <= 0 or reserve < 1 or margin < 0:
        raise ValueError("invalid context allocation")
    fixed = count_tokens(
        model=model,
        system=system,
        messages=[{"role": "user", "content": question}],
        tools=tools,
    )
    available = window - reserve - margin - fixed
    if available < 0:
        raise ValueError(
            "fixed request and output reserve exceed window"
        )
    return available
