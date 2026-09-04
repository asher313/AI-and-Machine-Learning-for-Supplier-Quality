# sqm_ai/gateway/audit.py
import hashlib
import json

import psycopg

INSERT = """
INSERT INTO llm_audit (
  trace_id, ts, user_id, tool_name, model, enclave,
  prompt_hash, prompt_tokens, response_hash, response_tokens,
  pre_warnings, post_warnings, blocked, block_codes,
  latency_ms, cost_usd
) VALUES (
  %(trace_id)s, now(), %(user_id)s, %(tool_name)s, %(model)s,
  %(enclave)s, %(prompt_hash)s, %(prompt_tokens)s,
  %(response_hash)s, %(response_tokens)s, %(pre)s, %(post)s,
  %(blocked)s, %(block_codes)s, %(latency_ms)s, %(cost_usd)s
)
"""


def sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditWriter:
    def __init__(self, dsn: str):
        self.conn = psycopg.connect(dsn, autocommit=True)

    def write(self, row: dict) -> None:
        row = dict(row)
        row["pre"] = json.dumps(row.pop("pre_warnings", []))
        row["post"] = json.dumps(row.pop("post_warnings", []))
        self.conn.execute(INSERT, row)
