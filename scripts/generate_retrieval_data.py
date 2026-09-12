"""Generate a small fictional corpus; no licensed standard text or real records."""

import argparse
import json
from pathlib import Path


def generate():
    documents = []
    for program in ("NL-KESTREL", "NL-MERLIN"):
        for i in range(1, 7):
            documents.append(
                {
                    "document_id": f"SYN-{program}-{i}",
                    "document_revision": "v1",
                    "source_type": "manual" if i < 4 else "car",
                    "structured": True,
                    "text": f"Fictional teaching record, not an actual quality requirement.\n8.4.{i} Example procedure {i}\nFor synthetic part 774{i}-B in {program}, the scenario requires an engineer to review the recorded measurement before choosing a disposition. No automatic release is allowed. The case label is example-{i}.",
                    "metadata": {
                        "visibility": "program",
                        "program": program,
                        "synthetic": True,
                    },
                    "valid_from": "2025-01-01",
                    "valid_to": None,
                }
            )
    documents.append(
        {
            "document_id": "SYN-SHARED",
            "document_revision": "v1",
            "source_type": "audit",
            "structured": False,
            "text": "This shared fictional training guide requires human review of every suggested disposition. It does not replace an approved procedure.",
            "metadata": {
                "visibility": "shared",
                "synthetic": True,
            },
            "valid_from": "2025-01-01",
            "valid_to": None,
        }
    )
    return {
        "synthetic": True,
        "purpose": "Retrieval plumbing and access tests, not measured semantic quality",
        "documents": documents,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/retrieval/documents.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(generate(), stream, indent=2)
        stream.write("\n")
    print(f"Wrote 13 fictional source documents to {args.output}")


if __name__ == "__main__":
    main()
