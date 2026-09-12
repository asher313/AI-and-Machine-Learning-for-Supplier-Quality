"""Ragas 0.4 collections API with explicit scoring and embedding dependencies."""

import asyncio
import math

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

from sqm_ai.llm import (
    MODELS,
    get_client,
    with_retry,
    awith_retry,
    parsed_response,
    log_usage,
    request_options,
)
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from sqm_ai.retrieval.embed import embed_query


class AnthropicScorer(InstructorBaseRagasLLM):
    """Implement Ragas' structured interface with the native SDK adapter."""

    def __init__(self, *, sync_client=None, async_client=None):
        self.sync_client = sync_client
        self.async_client = async_client
        self.model = MODELS["standard"]

    def _request(self, prompt, response_model):
        messages = [{"role": "user", "content": prompt}]
        return dict(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            output_format=response_model,
            **request_options(self.model),
        )

    def _result(self, response):
        log_usage(response, tool="assistant", stage="evaluation")
        return parsed_response(response)

    def generate(self, prompt, response_model):
        client = self.sync_client or get_client()
        request = self._request(prompt, response_model)
        count = with_retry(
            client.messages.count_tokens,
            model=self.model,
            messages=request["messages"],
        )
        if count.input_tokens + 3072 > 200000:
            raise ValueError("evaluation context budget exceeded")
        return self._result(
            with_retry(client.messages.parse, **request)
        )

    async def agenerate(self, prompt, response_model):
        client = self.async_client or get_client(
            asynchronous=True
        )
        request = self._request(prompt, response_model)
        count = await awith_retry(
            client.messages.count_tokens,
            model=self.model,
            messages=request["messages"],
        )
        if count.input_tokens + 3072 > 200000:
            raise ValueError("evaluation context budget exceeded")
        response = await awith_retry(
            client.messages.parse, **request
        )
        return self._result(response)


class LocalQueryEmbeddings(BaseRagasEmbedding):
    """Compare original/generated questions in the same pinned BGE query space."""

    def embed_text(self, text, **kwargs):
        return embed_query(text).tolist()

    async def aembed_text(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_text, text)


async def evaluate_rows(rows, *, llm, embeddings):
    scorers = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(
            llm=llm, embeddings=embeddings
        ),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    results = []
    for row in rows:
        args = {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["contexts"],
            "reference": row["reference"],
        }
        values = {}
        selected = {
            "faithfulness": [
                "user_input",
                "response",
                "retrieved_contexts",
            ],
            "answer_relevancy": ["user_input", "response"],
            "context_precision": [
                "user_input",
                "reference",
                "retrieved_contexts",
            ],
            "context_recall": [
                "user_input",
                "reference",
                "retrieved_contexts",
            ],
        }
        for name, scorer in scorers.items():
            result = await scorer.ascore(
                **{k: args[k] for k in selected[name]}
            )
            value = float(result.value)
            values[name] = value if math.isfinite(value) else None
        results.append({"id": row["id"], **values})
    return results
