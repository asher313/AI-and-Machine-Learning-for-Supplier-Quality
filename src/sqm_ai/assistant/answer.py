# Chapter 18 — 18.4 Three Layers Against Hallucination
# src/sqm_ai/assistant/answer.py
from dataclasses import dataclass

from sqm_ai.assistant.prompts import REFUSAL, SYSTEM
from sqm_ai.llm import MODELS, client, log_usage, with_retry


@dataclass
class Answer:
    text: str
    chunks: list[dict]    # the chunk rows, in [N] order
    refused: bool
    warnings: list[str]


def build_context(chunks: list[dict]) -> str:
    """Number the chunks; [N] in the answer is chunks[N-1]."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        ref = c["clause"] or c["document_id"]
        parts.append(
            f"[{i}] source={c['source_type']} ref={ref}\n"
            f"{c['content']}"
        )
    return "\n\n".join(parts)


def draft_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    response = with_retry(
        client.messages.create,
        model=MODELS["frontier"],
        max_tokens=900,
        system=[
            {"type": "text", "text": SYSTEM,
             "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{
            "role": "user",
            "content": (
                f"Documents:\n{context}\n\n"
                f"Question: {question}"
            ),
        }],
    )
    log_usage(response, tool="assistant", stage="draft")
    return "".join(
        b.text for b in response.content if b.type == "text"
    )


# Chapter 18 — 18.4 Three Layers Against Hallucination (continued)
def answer_question(question, chunks: list[dict]) -> Answer:
    """Draft, validate, and decide what the user sees."""
    if not chunks:
        return Answer(REFUSAL, [], True, ["no chunks"])

    text = draft_answer(question, chunks)
    if text.strip() == REFUSAL:
        return Answer(REFUSAL, chunks, True, [])

    context = build_context(chunks)
    warnings = structural_check(text, len(chunks))
    report = check_citations(text, context)
    bad = [c for c in report.checks if not c.supported]
    warnings += [
        f"[{c.citation}] unsupported: {c.reason}" for c in bad
    ]
    return Answer(text, chunks, False, warnings)
