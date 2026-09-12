"""Exercise Ragas -> native Anthropic structured output with a local HTTP mock only."""

import asyncio
import json

import pytest


def test_four_ragas_metrics_use_explicit_adapter_without_network(
    monkeypatch,
):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    import anthropic
    import httpx2 as httpx
    from ragas.embeddings.base import BaseRagasEmbedding

    from sqm_ai.assistant.evaluation import (
        evaluate_rows,
        AnthropicScorer,
    )
    from sqm_ai.llm import MODELS

    calls = []
    statement = "An engineer reviews the disposition."
    payloads = {
        "StatementGeneratorOutput": {"statements": [statement]},
        "NLIStatementOutput": {
            "statements": [
                {
                    "statement": statement,
                    "reason": "Explicitly present in the context.",
                    "verdict": 1,
                }
            ]
        },
        "AnswerRelevanceOutput": {
            "question": "Who reviews the disposition?",
            "noncommittal": 0,
        },
        "ContextPrecisionOutput": {
            "reason": "The context provides the answer.",
            "verdict": 1,
        },
        "ContextRecallOutput": {
            "classifications": [
                {
                    "statement": statement,
                    "reason": "Supported by the retrieved text.",
                    "attributed": 1,
                }
            ]
        },
    }

    async def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("count_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        calls.append(body)
        assert body["model"] == MODELS["standard"]
        assert "temperature" not in body and "top_p" not in body
        name = body["output_config"]["format"]["schema"]["title"]
        assert name in payloads
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": MODELS["standard"],
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payloads[name]),
                    }
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 10,
                },
            },
        )

    class Embeddings(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return [1.0, 0.0]

        async def aembed_text(self, text, **kwargs):
            return self.embed_text(text)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            client = anthropic.AsyncAnthropic(
                api_key="test-only",
                http_client=http,
                max_retries=0,
            )
            llm = AnthropicScorer(async_client=client)
            rows = [
                {
                    "id": "synthetic",
                    "question": "Who reviews the disposition?",
                    "answer": statement,
                    "contexts": [statement],
                    "reference": statement,
                }
            ]
            return await evaluate_rows(
                rows, llm=llm, embeddings=Embeddings()
            )

    result = asyncio.run(run())
    assert (
        len(calls) == 7
    )  # 2 faithfulness + 3 relevance + 1 precision + 1 recall
    assert result[0]["faithfulness"] == 1
    assert result[0]["context_recall"] == 1
    assert result[0]["context_precision"] == pytest.approx(1)
    assert result[0]["answer_relevancy"] == 1
