import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from sqm_ai import llm


def test_model_options_and_import_without_credentials():
    assert llm.request_options(llm.MODELS["fast"]) == {}
    assert (
        llm.request_options(llm.MODELS["standard"])[
            "output_config"
        ]["effort"]
        == "low"
    )


def test_cache_ttl_cost_and_unknown_tier():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        cache_creation_input_tokens=3000,
        cache_read_input_tokens=500,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=1000,
            ephemeral_1h_input_tokens=2000,
        ),
        service_tier="standard",
    )
    assert llm.estimate_cost(
        "claude-sonnet-5", usage
    ) == pytest.approx(0.0136)
    usage.service_tier = "priority"
    with pytest.raises(ValueError, match="modifier"):
        llm.estimate_cost("claude-sonnet-5", usage)


def test_retry_headers_and_nonretryable_errors(monkeypatch):
    delays = []
    monkeypatch.setattr(llm.time, "sleep", delays.append)
    response = httpx.Response(
        429,
        headers={"retry-after-ms": "250"},
        request=httpx.Request("POST", "https://example.invalid"),
    )
    error = anthropic.RateLimitError(
        "test", response=response, body=None
    )
    calls = 0

    def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return "ok"

    assert llm.with_retry(request, base=0.01) == "ok"
    assert calls == 2 and delays == [0.25]
    bad = anthropic.BadRequestError(
        "test",
        response=httpx.Response(400, request=response.request),
        body=None,
    )
    with pytest.raises(anthropic.BadRequestError):
        llm.with_retry(lambda: (_ for _ in ()).throw(bad))
    assert delays == [0.25]
    with pytest.raises(ValueError):
        llm.with_retry(request, attempts=0)


def test_async_retry_and_text_stop_reason(monkeypatch):
    async def immediate(delay):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", immediate)
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise anthropic.APIConnectionError(
                request=httpx.Request(
                    "POST", "https://example.invalid"
                )
            )
        return 7

    assert asyncio.run(llm.awith_retry(request)) == 7
    response = SimpleNamespace(
        stop_reason="max_tokens", content=[]
    )
    with pytest.raises(ValueError, match="incomplete"):
        llm.text_response(response)


def test_budget_counts_actual_system_and_question(monkeypatch):
    captured = {}

    def count(**kwargs):
        captured.update(kwargs)
        return 100

    monkeypatch.setattr(llm, "count_tokens", count)
    assert (
        llm.context_budget(
            "system",
            "question",
            window=1000,
            reserve=200,
            margin=50,
        )
        == 650
    )
    assert captured["system"] == "system"
    assert captured["messages"] == [
        {"role": "user", "content": "question"}
    ]
    with pytest.raises(ValueError):
        llm.context_budget("s", "q", window=100, reserve=50)
    assert (
        llm.pack_context([("large", 90), ("small", 5)], 50) == []
    )
