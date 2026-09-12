from types import SimpleNamespace as NS

import numpy as np
import pytest

from sqm_ai.retrieval import embed
from sqm_ai.retrieval.chunking import (
    clause_chunks,
    recursive_chunks,
)
from sqm_ai.retrieval.fuse import rrf, sparse_rank
from sqm_ai.retrieval.rerank import rerank


def test_embedding_space_prefix_and_no_silent_truncation():
    seen = []

    class Model:
        max_seq_length = 50

        def tokenizer(self, t, truncation):
            assert truncation is False
            return {"input_ids": t.split()}

        def encode(self, texts, **kwargs):
            seen.extend(texts)
            out = np.zeros((len(texts), 1024), dtype=np.float32)
            out[:, 0] = 1
            return out

    model = Model()
    embed.embed_query("part 7741-B", model=model)
    embed.embed_corpus(["part 7741-B"], model=model)
    assert seen == [
        embed.QUERY_PREFIX + "part 7741-B",
        "part 7741-B",
    ]
    with pytest.raises(ValueError, match="too long"):
        embed.embed_corpus(["word " * 51], model=model)


def test_rank_fusion_duplicate_and_one_based_ranks():
    assert rrf([[2, 2, 1], [1]]) == [1, 2]
    assert rrf([[2], [1]]) == [1, 2]  # stable deterministic tie
    with pytest.raises(ValueError):
        rrf([[1]], k=0)
    assert sparse_rank(
        "7741-B",
        [
            {"id": 1, "content": "part 7741-B."},
            {"id": 2, "content": "part 7741-C"},
        ],
    ) == [1]


def test_chunker_keeps_preamble_and_long_clause_metadata():
    text = (
        "Preamble retained.\n8.4.2 Supplier controls\n"
        + "A long fictional requirement. " * 100
    )
    chunks = clause_chunks(text, "doc", chunk_size=200)
    assert chunks[0]["content"].startswith("Preamble")
    clauses = [c for c in chunks if c["clause"] == "8.4.2"]
    assert len(clauses) > 1 and all(
        c["content"].startswith("8.4.2") for c in clauses
    )
    with pytest.raises(ValueError):
        recursive_chunks("text", "doc", overlap=800)


def test_rerank_handles_empty_and_does_not_load_weights():
    assert rerank("q", []) == []
    rows = [
        {"id": 1, "content": "irrelevant"},
        {"id": 2, "content": "relevant"},
    ]
    fake = NS(predict=lambda pairs: np.array([-0.2, 0.9]))
    assert rerank("q", rows, k=1, model=fake)[0]["id"] == 2
