"""Generate and run a synthetic gateway replay; no model API or cloud account."""

import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import json
import os
from pathlib import Path

import psycopg
from sqm_ai.gateway import Gateway
from sqm_ai.gateway.audit import (
    AuditWriter,
    EncryptedContentStore,
)
from sqm_ai.gateway.policy import Policy, GatewayUser
from sqm_ai.gateway.router import (
    BudgetLedger,
    Endpoint,
    ModelRouter,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        required=True,
        help="Disposable PostgreSQL database, not production",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(args.dsn) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS sqm")
        conn.execute(
            (
                Path(__file__).parents[1]
                / "src/sqm_ai/gateway/schema.sql"
            ).read_text()
        )
    # Ephemeral key: archive is verified in this process, unrecoverable afterwards.
    # Production must inject managed, retained keys and enforce storage access/retention.
    store = EncryptedContentStore(
        args.out / "archive",
        os.urandom(32),
        key_id="ephemeral-synthetic-only",
        allowed_levels={"open"},
    )
    calls = []

    def invoke(request):
        calls.append(request)
        return {
            "model": "synthetic-response-v1",
            "text": "Follow the fictional reviewed procedure [1].",
            "stop_reason": "end_turn",
            "charge": ".01",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }

    endpoint = Endpoint(
        "local-scripted",
        "synthetic-response-v1",
        frozenset({"open"}),
        invoke,
        lambda request: Decimal(".02"),
        lambda response: Decimal(response["charge"]),
    )
    policy = Policy(validate_citations=True)
    gateway = Gateway(
        policy,
        audit=AuditWriter(args.dsn),
        ledger=BudgetLedger(
            args.dsn,
            policy.monthly_budget_usd,
            policy.soft_alert_fraction,
        ),
        router=ModelRouter({("open", "standard"): endpoint}),
        content_store=store,
    )
    # Canonical 33/8 counts are constructed test cases, not observed detection rates.
    prompts = (
        ["Classify synthetic NCR; callback 918-555-0142"] * 33
        + ["Summarize CUI drawing note"] * 8
        + ["Summarize the synthetic procedure"]
    )
    results = []
    for prompt in prompts:
        result = gateway.complete(
            prompt=prompt,
            system="Use supplied fictional evidence.",
            user=GatewayUser("synthetic-reader"),
            tool_name="build6-replay",
            enclave="open",
            classification="open",
            sources=["Use the approved inspection procedure."],
        )
        results.append(asdict(result))
    counts = Counter(
        code for result in results for code in result["codes"]
    )
    assert (
        counts == {"GW-PII": 33, "GW-MARK": 8}
        and len(calls) == 1
        and not results[-1]["blocked"]
    )
    trace = results[-1]["trace_id"]
    with psycopg.connect(args.dsn) as conn:
        row = conn.execute(
            "SELECT record FROM sqm.llm_audit_events WHERE trace_id=%s AND event='completed'",
            (trace,),
        ).fetchone()[0]
    request = json.loads(
        store.get(
            row["request_ref"],
            trace_id=trace,
            kind="request",
            classification="open",
        )
    )
    assert request == calls[0]
    report = {
        "mode": "synthetic-scripted",
        "provider_api_calls": 0,
        "scripted_dispatches": len(calls),
        "constructed_blocks": dict(counts),
        "archive_roundtrip_verified": True,
        "archive_key": "ephemeral; not retained; encrypted files cannot be reopened after this run",
        "results": results,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "results"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
