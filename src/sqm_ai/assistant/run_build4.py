"""Offline software replay over the generated corpus, not a semantic-quality test."""

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sqm_ai.assistant.answer import answer_question
from sqm_ai.assistant.schema import Claim, Draft
from sqm_ai.assistant.validate import CitationReport, ClaimCheck
from sqm_ai.retrieval.fuse import rrf, sparse_rank
from sqm_ai.retrieval.synthetic import SPACE, hash_vectors


def run(bundle, question, program, asof):
    if (
        bundle.get("synthetic") is not True
        or bundle.get("embedding_mode") != "synthetic"
    ):
        raise ValueError(
            "generated lexical-hash index required for this replay"
        )
    rows = []
    for i, raw in enumerate(bundle["rows"], 1):
        row = dict(raw, id=i)
        meta = row["metadata"]
        visible = meta.get("visibility") == "shared" or (
            meta.get("visibility") == "program"
            and meta.get("program") == program
        )
        valid = row["valid_from"] <= asof and (
            row["valid_to"] is None or asof < row["valid_to"]
        )
        if visible and valid and row["embed_model"] == SPACE:
            rows.append(row)
    query = hash_vectors([question])[0]
    dense = sorted(
        rows,
        key=lambda row: (
            -float(np.asarray(row["embedding"]) @ query),
            row["id"],
        ),
    )
    ids = rrf(
        [
            [row["id"] for row in dense[:50]],
            sparse_rank(question, rows, 50),
        ]
    )[:8]
    lookup = {row["id"]: row for row in rows}
    chunks = [lookup[i] for i in ids]

    def drafter(q, sources, **kwargs):
        # Scripted extraction is for exercising the boundary only.
        return Draft(
            refused=False,
            claims=[
                Claim(text=sources[0]["content"], citations=[1])
            ],
        )

    def checker(q, draft, sources, **kwargs):
        return CitationReport(
            answers_question=True,
            checks=[
                ClaimCheck(
                    claim_id=1,
                    supported=draft.claims[0].text
                    in sources[0]["content"],
                    reason="Offline replay checks exact extraction only; question relevance is scripted.",
                )
            ],
        )

    result = answer_question(
        question,
        chunks,
        data_classification="synthetic",
        drafter=drafter,
        checker=checker,
    )
    return {
        "mode": "offline software replay",
        "neural_quality_evaluated": False,
        "api_calls": 0,
        "result": asdict(result),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/retrieval/index.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/build4/answer.json"),
    )
    parser.add_argument(
        "--question",
        default="Who reviews a suggested disposition?",
    )
    parser.add_argument("--program", default="NL-KESTREL")
    parser.add_argument("--asof", default="2026-09-01")
    args = parser.parse_args()
    dt.date.fromisoformat(args.asof)
    result = run(
        json.loads(args.index.read_text()),
        args.question,
        args.program,
        args.asof,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "result"}
            | {"status": result["result"]["status"]}
        )
    )


if __name__ == "__main__":
    main()
