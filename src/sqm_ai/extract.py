# src/sqm_ai/extract.py
from pathlib import Path

import pandas as pd
import structlog
import yaml
from sqlalchemy import create_engine, text

from sqm_ai.hana import hana_connection
from sqm_ai.settings import get_settings

log = structlog.get_logger()
MANIFEST = Path(__file__).parent / "sql" / "extract_manifest.yaml"
CHUNK = 100_000


class SchemaContractError(RuntimeError):
    """A HANA view no longer has the columns we rely on."""


def _check(name: str, got: list[str], want: list[str]) -> None:
    missing = [c for c in want if c not in got]
    if missing:
        raise SchemaContractError(f"{name} missing {missing}")


def extract_one(entry: dict, pg) -> int:
    """Copy one HANA view into Postgres. Returns row count."""
    src, dst = entry["source"], entry["target"]
    total = 0
    with hana_connection() as hana:
        reader = pd.read_sql(
            f"SELECT * FROM {src}", hana, chunksize=CHUNK
        )
        for i, chunk in enumerate(reader):
            if i == 0:
                _check(src, list(chunk.columns),
                       entry["required"])
            chunk.columns = [c.lower() for c in chunk.columns]
            chunk.to_sql(
                dst.split(".")[1],
                pg,
                schema=dst.split(".")[0],
                if_exists="replace" if i == 0 else "append",
                index=False,
                method="multi",
            )
            total += len(chunk)
    log.info("extracted", source=src, target=dst, rows=total)
    return total


def run() -> dict[str, int]:
    entries = yaml.safe_load(MANIFEST.read_text())
    pg = create_engine(get_settings().database_url)
    counts = {}
    with pg.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS sqm"))
    for entry in entries:
        counts[entry["target"]] = extract_one(entry, pg)
    log.info("extract_complete", tables=len(counts),
             rows=sum(counts.values()))
    return counts


if __name__ == "__main__":
    run()
