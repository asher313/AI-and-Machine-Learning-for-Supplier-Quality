"""Draft structured claims, validate complete coverage, then render a result."""

import json
from dataclasses import dataclass

import anthropic
from pydantic import ValidationError

from sqm_ai.assistant.prompts import (
    REFUSAL,
    SYSTEM,
    VERIFICATION_FAILURE,
)
from sqm_ai.assistant.schema import Draft
from sqm_ai.assistant.validate import (
    CitationReport,
    check_citations,
    report_problems,
    require_approved,
    structural_check,
)
from sqm_ai.llm import (
    MODELS,
    client,
    count_tokens,
    log_usage,
    parsed_response,
    request_options,
    with_retry,
)


@dataclass
class Answer:
    text: str
    chunks: list[dict]
    refused: bool
    warnings: list[str]
    status: str = "verified"


def source_view(chunks):
    keys = (
        "id",
        "source_type",
        "document_id",
        "document_revision",
        "clause",
        "content",
        "valid_from",
        "valid_to",
    )
    return [{k: c.get(k) for k in keys} for c in chunks]


def build_context(chunks):
    return json.dumps(
        [
            {"number": i, **c}
            for i, c in enumerate(source_view(chunks), 1)
        ],
        default=str,
    )


def draft_answer(
    question, chunks, *, data_classification="unknown"
):
    require_approved(data_classification)
    messages = [
        {
            "role": "user",
            "content": f"Sources: {build_context(chunks)}\nQuestion: {question}",
        }
    ]
    if (
        count_tokens(
            model=MODELS["frontier"],
            system=SYSTEM,
            messages=messages,
        )
        + 1600
        + 1024
        > 200000
    ):
        raise ValueError(
            "draft context budget exceeded; pack fewer or shorter sources"
        )
    response = with_retry(
        client.messages.parse,
        model=MODELS["frontier"],
        max_tokens=1600,
        system=SYSTEM,
        messages=messages,
        output_format=Draft,
        **request_options(MODELS["frontier"]),
    )
    log_usage(response, tool="assistant", stage="draft")
    return parsed_response(response)


def answer_question(
    question,
    chunks,
    *,
    data_classification="unknown",
    drafter=None,
    checker=None,
):
    """Input chunks must already be authorized; injected adapters enforce their boundary."""
    if not chunks:
        return Answer(
            REFUSAL,
            [],
            True,
            ["no authorized sources"],
            "refused",
        )
    chunks = source_view(chunks)
    try:
        draft = (drafter or draft_answer)(
            question,
            chunks,
            data_classification=data_classification,
        )
        draft = Draft.model_validate(draft)
        if draft.refused:
            return Answer(REFUSAL, chunks, True, [], "refused")
        problems = structural_check(draft, len(chunks))
        if not problems:
            report = (checker or check_citations)(
                question,
                draft,
                chunks,
                data_classification=data_classification,
            )
            problems = report_problems(
                CitationReport.model_validate(report), draft
            )
        if problems:
            return Answer(
                VERIFICATION_FAILURE,
                chunks,
                True,
                problems,
                "verification_failed",
            )
        text = "\n".join(
            c.text + " " + " ".join(f"[{n}]" for n in c.citations)
            for c in draft.claims
        )
        return Answer(text, chunks, False, [], "verified")
    except (
        anthropic.APIError,
        ValidationError,
        ValueError,
    ) as exc:
        return Answer(
            VERIFICATION_FAILURE,
            chunks,
            True,
            [type(exc).__name__],
            "verification_failed",
        )
