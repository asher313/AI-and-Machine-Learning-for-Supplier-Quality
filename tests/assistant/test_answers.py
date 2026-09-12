from sqm_ai.assistant.answer import answer_question
from sqm_ai.assistant.schema import Claim, Draft
from sqm_ai.assistant.validate import CitationReport, ClaimCheck

CHUNKS = [
    {
        "id": 1,
        "document_id": "SYN-1",
        "document_revision": "v1",
        "source_type": "manual",
        "clause": "1",
        "content": "Fictional rule: an engineer reviews a suggested disposition.",
    }
]


def draft(*args, **kwargs):
    return Draft(
        refused=False,
        claims=[
            Claim(
                text="An engineer reviews the suggestion.",
                citations=[1],
            ),
            Claim(
                text="A human retains disposition authority.",
                citations=[1],
            ),
        ],
    )


def report(*args, **kwargs):
    return CitationReport(
        answers_question=True,
        checks=[
            ClaimCheck(
                claim_id=i,
                supported=True,
                reason="The cited source supports the stated claim.",
            )
            for i in [1, 2]
        ],
    )


def test_same_citation_requires_each_claim_and_empty_report_fails():
    ok = answer_question(
        "Who reviews?", CHUNKS, drafter=draft, checker=report
    )
    assert ok.status == "verified" and ok.text.count("[1]") == 2
    for ids in [[], [1], [1, 1], [1, 2, 3]]:

        def checker(*args, ids=ids, **kwargs):
            return CitationReport(
                answers_question=True,
                checks=[
                    ClaimCheck(
                        claim_id=i,
                        supported=True,
                        reason="The citation appears to support this claim.",
                    )
                    for i in ids
                ],
            )

        answer = answer_question(
            "Who reviews?", CHUNKS, drafter=draft, checker=checker
        )
        assert (
            answer.status == "verification_failed"
            and "An engineer reviews" not in answer.text
        )


def test_bad_marker_rejects_before_checker():
    def bad(*args, **kwargs):
        return Draft(
            refused=False,
            claims=[Claim(text="A source claim.", citations=[2])],
        )

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "invalid citation must not reach checker"
        )

    assert (
        answer_question(
            "q", CHUNKS, drafter=bad, checker=unexpected
        ).status
        == "verification_failed"
    )


def test_unsupported_and_irrelevant_answers_are_not_displayed():
    def unsupported(*args, **kwargs):
        result = report()
        result.checks[1].supported = False
        return result

    def irrelevant(*args, **kwargs):
        result = report()
        result.answers_question = False
        return result

    for checker in [unsupported, irrelevant]:
        assert (
            answer_question(
                "q", CHUNKS, drafter=draft, checker=checker
            ).status
            == "verification_failed"
        )


def test_refusal_and_verification_failure_remain_distinct():
    assert answer_question("q", []).status == "refused"

    def refused(*args, **kwargs):
        return Draft(refused=True, claims=[])

    assert (
        answer_question("q", CHUNKS, drafter=refused).status
        == "refused"
    )
    # Unknown data classification cannot silently enter the direct API path.
    assert (
        answer_question("q", CHUNKS).status
        == "verification_failed"
    )
