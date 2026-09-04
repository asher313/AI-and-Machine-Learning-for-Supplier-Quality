# Chapter 2 — Generators
import json
from collections.abc import Iterator
from pathlib import Path


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:            # one line in memory at a time
            yield json.loads(line)


def batched(items: Iterator[dict], size: int) -> Iterator[list]:
    batch: list[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


for batch in batched(read_jsonl(Path("ncrs.jsonl")), 32):
    ...                           # 32 NCRs at a time
