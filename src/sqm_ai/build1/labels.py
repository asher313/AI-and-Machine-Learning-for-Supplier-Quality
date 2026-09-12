"""Read forward targets with an explicit coverage watermark."""

from pathlib import Path

import pandas as pd
from sqlalchemy import text


def build_labels(engine, data_complete_through):
    sql = (
        Path(__file__).parent / "sql" / "labels.sql"
    ).read_text()
    return pd.read_sql_query(
        text(sql),
        engine,
        params={
            "data_complete_through": data_complete_through,
        },
        parse_dates=[
            "month",
            "label_end",
            "data_complete_through",
        ],
    )
