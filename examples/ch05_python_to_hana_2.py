# Chapter 5 — 5.4 Python to HANA
from sqlalchemy import create_engine

engine = create_engine(
    "hana+hdbcli://sqm_readonly:PASSWORD@"
    "hana.northlake.internal:30015"
)
