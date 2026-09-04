# src/sqm_ai/assistant/versioning.py
import datetime as dt

import psycopg


def supersede(
    conn: psycopg.Connection, document_id: str,
    effective: dt.date,
) -> int:
    """Close every open chunk of one document."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE sqm.doc_chunks
               SET valid_to = %s
             WHERE document_id = %s
               AND valid_to IS NULL
            """,
            (effective, document_id),
        )
        return cur.rowcount
