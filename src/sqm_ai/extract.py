# src/sqm_ai/extract.py
from pathlib import Path
import re

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
    if not re.fullmatch(r"[A-Za-z_][\w]*\.[A-Za-z_][\w]*", src):
        raise ValueError("source must be a schema.object name")
    total = 0
    with hana_connection() as hana:
        cursor = hana.cursor()
        try:
            cursor.execute(f"SELECT * FROM {src}")
            columns = [d[0].lower() for d in cursor.description]
            _check(src, columns, entry["required"])
            if len(set(columns)) != len(columns):
                raise SchemaContractError("duplicate column names")
            first = True
            while True:
                rows = cursor.fetchmany(CHUNK)
                if not rows and not first:
                    break
                chunk = pd.DataFrame(rows, columns=columns)
                chunk.to_sql(
                    dst.split(".")[1], pg,
                    schema=dst.split(".")[0],
                    if_exists="replace" if first else "append",
                    index=False, method="multi", chunksize=1000,
                )
                first = False
                total += len(chunk)
                if not rows:
                    break
        finally:
            cursor.close()
    log.info("extracted", source=src, target=dst, rows=total)
    return total


def run() -> dict[str, int]:
    entries = yaml.safe_load(MANIFEST.read_text())
    pg = create_engine(get_settings().database_url)
    counts = {}
    try:
        with pg.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS sqm"))
            for entry in entries:
                counts[entry["target"]] = extract_one(entry, conn)
    finally:
        pg.dispose()
    log.info("extract_complete", tables=len(counts),
             rows=sum(counts.values()))
    return counts


if __name__ == "__main__":
    run()
