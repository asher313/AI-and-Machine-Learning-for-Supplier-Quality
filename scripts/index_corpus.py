"""Index generated sources; optional explicit PostgreSQL insertion, no implicit writes."""

import argparse
import json
from pathlib import Path

from sqm_ai.retrieval.chunking import (
    clause_chunks,
    recursive_chunks,
)
from sqm_ai.retrieval.embed import EMBED_SPACE, embed_corpus
from sqm_ai.retrieval.store import connect, insert_chunks
from sqm_ai.retrieval.synthetic import SPACE, hash_vectors


def build_rows(documents, *, embedding="synthetic"):
    if embedding not in {"synthetic", "bge"}:
        raise ValueError("unknown embedding mode")
    rows = []
    for doc in documents:
        chunks = (
            clause_chunks
            if doc["structured"]
            else recursive_chunks
        )(doc["text"], doc["document_id"])
        texts = [c["content"] for c in chunks]
        vectors = (
            hash_vectors(texts)
            if embedding == "synthetic"
            else embed_corpus(texts)
        )
        for i, (chunk, vector) in enumerate(
            zip(chunks, vectors, strict=True)
        ):
            rows.append(
                {
                    "document_id": doc["document_id"],
                    "document_revision": doc["document_revision"],
                    "source_type": doc["source_type"],
                    "chunk_index": i,
                    "clause": chunk["clause"],
                    "content": chunk["content"],
                    "token_count": len(chunk["content"].encode()),
                    "token_model": "utf8-byte-estimate-not-provider-count",
                    "embedding": vector.tolist(),
                    "embed_model": SPACE
                    if embedding == "synthetic"
                    else EMBED_SPACE,
                    "metadata": doc["metadata"],
                    "valid_from": doc["valid_from"],
                    "valid_to": doc.get("valid_to"),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/retrieval/documents.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval/index.json"),
    )
    parser.add_argument(
        "--embedding",
        choices=["synthetic", "bge"],
        default="synthetic",
    )
    parser.add_argument(
        "--database-url",
        help="Explicit PostgreSQL destination with reviewed schema already installed",
    )
    args = parser.parse_args()
    bundle = json.loads(args.data.read_text())
    if bundle.get("synthetic") is not True:
        raise ValueError(
            "this teaching CLI requires generated synthetic sources"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = build_rows(
        bundle["documents"], embedding=args.embedding
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "synthetic": True,
                "embedding_mode": args.embedding,
                "rows": rows,
            },
            stream,
        )
    if args.database_url:
        with connect(args.database_url) as conn:
            insert_chunks(conn, rows)
    print(
        json.dumps(
            {
                "documents": len(bundle["documents"]),
                "chunks": len(rows),
                "embedding_mode": args.embedding,
                "database_written": bool(args.database_url),
                "neural_quality_evaluated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
