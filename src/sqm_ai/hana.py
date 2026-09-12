# src/sqm_ai/hana.py
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd
import structlog
from hdbcli import dbapi

from sqm_ai.settings import get_settings

log = structlog.get_logger()


@contextmanager
def hana_connection() -> Iterator[dbapi.Connection]:
    """Open a read-only HANA connection; always close it."""
    s = get_settings()
    conn = dbapi.connect(
        address=s.hana_host,
        port=s.hana_port,
        user=s.hana_user,
        password=s.hana_password.get_secret_value(),
        encrypt=True,
        sslValidateCertificate=True,
    )
    try:
        yield conn
    finally:
        conn.close()


def read_hana(
    sql: str, params: tuple[Any, ...] = ()
) -> pd.DataFrame:
    """Run a parameterized SELECT and return a DataFrame."""
    with hana_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
            columns = [d[0].lower() for d in cursor.description]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)
        finally:
            cursor.close()
    log.info("hana_read", rows=len(df), cols=len(df.columns))
    return df
