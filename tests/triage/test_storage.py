from importlib.resources import files

from sqlalchemy import text

from sqm_ai.triage.pipeline import triage_one
from sqm_ai.triage.schema import TriageResult
from sqm_ai.triage.storage import write_decision


def test_append_attempts_and_preserve_failed_result(db):
    # db fixture owns an isolated UUID database; no application tables are touched.
    script = (
        files("sqm_ai.triage").joinpath("schema.sql").read_text()
    )
    with db.begin() as conn:
        for statement in script.split(";"):
            if (
                statement.strip()
                and not statement.strip().startswith("-- Probe")
            ):
                conn.execute(text(statement))
        ncr = {
            "ncr_id": "SYN-1",
            "part_number": "7741-B",
            "quantity": 1,
            "defect_description": "A bracket has a dimensional discrepancy.",
            "supplier_shortlist": ["S-0417"],
            "data_classification": "restricted",
        }
        failure = triage_one(ncr)
        write_decision(
            conn,
            failure,
            source_commit_sha="a" * 40,
            classifier_version="test",
        )
        row = conn.execute(
            text(
                "SELECT category,confidence,routed FROM sqm.ncr_classifications"
            )
        ).one()
        assert row == (None, None, "review")

        # Re-triage appends another attempt for the same NCR identity.
        def stub(*args, **kwargs):
            return TriageResult(
                category="dimensional",
                severity=3,
                supplier_id="S-0417",
                suggested_disposition="rework",
                confidence=0.9,
                reasoning="A scripted response for database contract testing.",
            ), "offline"

        success = triage_one(ncr, classifier=stub)
        write_decision(
            conn,
            success,
            source_commit_sha="b" * 40,
            classifier_version="test",
        )
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM sqm.ncr_classifications"
                )
            ).scalar_one()
            == 2
        )
