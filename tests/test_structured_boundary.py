import asyncio
from types import SimpleNamespace as NS

import numpy as np
import pytest
from pydantic import ValidationError

from sqm_ai import conversation, structured
from sqm_ai.semantic_cache import CacheScope, SemanticCache

VALID = {
    "category": "dimensional",
    "severity": 3,
    "suggested_disposition": "rework",
    "confidence": 0.8,
    "reasoning": "The hole position exceeds the stated tolerance.",
}


def test_strict_tool_schema_and_response(monkeypatch):
    schema = structured.CLASSIFY_TOOL["input_schema"]
    assert schema["additionalProperties"] is False
    assert "minimum" not in schema["properties"]["severity"]
    response = NS(
        stop_reason="tool_use",
        content=[
            NS(
                type="tool_use",
                name="record_classification",
                input=VALID,
            )
        ],
    )
    monkeypatch.setattr(
        structured,
        "client",
        NS(messages=NS(create=lambda **kwargs: response)),
    )
    assert (
        structured.classify_via_tool("fictional NCR").severity
        == 3
    )
    response.content[0].input = {**VALID, "severity": 7}
    with pytest.raises(ValidationError):
        structured.classify_via_tool("fictional NCR")
    response.content[0].input = VALID
    response.content[0].name = "execute_disposition"
    with pytest.raises(ValueError, match="one complete"):
        structured.classify_via_tool("fictional NCR")


def test_async_preserves_order_and_failures_and_bounds_tasks(
    monkeypatch,
):
    active = maximum = 0

    async def fake(description):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        await asyncio.sleep(0)
        active -= 1
        if description == 3:
            raise ValueError("invalid record")
        return description

    monkeypatch.setattr(structured, "classify_one", fake)
    results = asyncio.run(
        structured.classify_many(list(range(10)), limit=2)
    )
    assert (
        maximum == 2
        and results[8] == 8
        and isinstance(results[3], ValueError)
    )
    with pytest.raises(ValueError):
        asyncio.run(structured.classify_many([], limit=0))


def test_conversation_does_not_commit_failed_turn(monkeypatch):
    monkeypatch.setattr(
        conversation, "count_tokens", lambda **kwargs: 20
    )
    response = NS(stop_reason="max_tokens", content=[])
    monkeypatch.setattr(
        conversation,
        "client",
        NS(messages=NS(create=lambda **kwargs: response)),
    )
    chat = conversation.Conversation("system")
    with pytest.raises(ValueError):
        chat.send("question")
    assert chat.messages == []

    class Block:
        type = "text"
        text = "answer"

        def model_dump(self, **kwargs):
            return {"type": self.type, "text": self.text}

    response.stop_reason, response.content = "end_turn", [Block()]
    assert chat.send("question") == "answer"
    assert len(chat.messages) == 2
    assert chat.messages[1]["content"] == [
        {"type": "text", "text": "answer"}
    ]


def test_cache_requires_scope_revision_ttl_and_semantic_verification():
    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array(
                [[1.0, 0.0] for _ in texts]
            )  # deliberately indistinguishable

    now = [0.0]
    cache = SemanticCache(
        Embedder(), ttl=10, capacity=2, clock=lambda: now[0]
    )
    scope = CacheScope(
        "role-a", "doc-v1", "prompt-v1", "embed-v1"
    )
    cache.put("clause 8.4.2", {"answer": "A"}, scope=scope)
    assert cache.get("clause 8.4.3", scope=scope) is None
    assert (
        cache.get(
            "clause 8.4.2",
            scope=CacheScope(
                "role-b", "doc-v1", "prompt-v1", "embed-v1"
            ),
        )
        is None
    )
    assert (
        cache.get(
            "clause 8.4.2",
            scope=CacheScope(
                "role-a", "doc-v2", "prompt-v1", "embed-v1"
            ),
        )
        is None
    )
    answer = cache.get("clause 8.4.2", scope=scope)
    answer["answer"] = "corrupted"
    assert cache.get("clause 8.4.2", scope=scope)["answer"] == "A"
    now[0] = 11
    assert cache.get("clause 8.4.2", scope=scope) is None


def test_cascade_omits_unsupported_effort_and_preserves_review(
    monkeypatch,
):
    seen = []

    def parse(**kwargs):
        seen.append(kwargs)
        return NS(
            stop_reason="end_turn",
            parsed_output=structured.NCRClassification(**VALID),
        )

    monkeypatch.setattr(
        structured, "client", NS(messages=NS(parse=parse))
    )
    result = structured.classify_cascaded("fictional")
    assert len(seen) == 3 and "output_config" not in seen[0]
    assert result.needs_review is True
