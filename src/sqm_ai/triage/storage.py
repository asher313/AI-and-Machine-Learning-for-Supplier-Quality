"""Append validated decision attempts using an existing SQLAlchemy transaction."""

import re

from sqlalchemy import text


def write_decision(
    conn, decision, *, source_commit_sha, classifier_version
):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit_sha):
        raise ValueError("full source commit SHA required")
    row = decision.model_dump(exclude={"result"})
    keys = [
        "category",
        "severity",
        "supplier_id",
        "suggested_disposition",
        "confidence",
        "reasoning",
    ]
    row.update(
        decision.result.model_dump()
        if decision.result
        else dict.fromkeys(keys)
    )
    row.update(
        source_commit_sha=source_commit_sha,
        classifier_version=classifier_version,
    )
    columns = list(row)
    values = [
        f"CAST(:{k} AS uuid[])" if k == "trace_ids" else f":{k}"
        for k in columns
    ]
    conn.execute(
        text(
            f"INSERT INTO sqm.ncr_classifications ({','.join(columns)}) VALUES ({','.join(values)})"
        ),
        row,
    )
