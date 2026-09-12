"""Explicit live Ragas evaluation of approved saved answers; paid calls required."""

import argparse
import asyncio
import json
import os
from pathlib import Path


async def run(rows):
    # Set before importing Ragas; operational telemetry is unnecessary for this recipe.
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"

    from sqm_ai.assistant.evaluation import (
        LocalQueryEmbeddings,
        AnthropicScorer,
        evaluate_rows,
    )
    from sqm_ai.assistant.validate import require_approved

    for row in rows:
        require_approved(
            row.get("data_classification", "unknown")
        )
    llm = AnthropicScorer()
    return await evaluate_rows(
        rows, llm=llm, embeddings=LocalQueryEmbeddings()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Explicitly enable paid provider calls and local embedding model loading",
    )
    args = parser.parse_args()
    if not args.run_live:
        parser.error(
            "--run-live is required; offline software validation uses the mocked tests"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = json.loads(args.input.read_text())
    if not rows:
        raise ValueError("nonempty evaluation rows required")
    result = asyncio.run(run(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {"ragas_version": "0.4.3", "results": result},
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
