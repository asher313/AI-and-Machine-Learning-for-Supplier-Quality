# Chapter 18 — 18.4 Three Layers Against Hallucination
# src/sqm_ai/assistant/validate.py
import re

from pydantic import BaseModel, Field

from sqm_ai.llm import MODELS, client, log_usage, with_retry

CITE = re.compile(r"\[(\d+)\]")


def cited_numbers(answer: str) -> list[int]:
    """Every [N] marker in the answer, in order, deduped."""
    seen, out = set(), []
    for m in CITE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def structural_check(answer: str, n_chunks: int) -> list[str]:
    """Cheap checks that need no model call."""
    nums = cited_numbers(answer)
    problems = []
    for n in nums:
        if n < 1 or n > n_chunks:
            problems.append(f"citation [{n}] does not exist")
    if not nums:
        problems.append("answer contains no citation")
    return problems


# Chapter 18 — 18.4 Three Layers Against Hallucination (continued)
class CitationCheck(BaseModel):
    citation: int = Field(ge=1)
    supported: bool
    reason: str = Field(min_length=10, max_length=300)


class CitationReport(BaseModel):
    checks: list[CitationCheck]


CHECK_SYSTEM = """\
You verify citations. For each [N] citation in the answer,
decide whether the claim it marks is stated or directly
implied by document N. Judge only support, never truth: a
correct statement that document N does not contain is NOT
supported. Give one check per distinct citation number.
"""


def check_citations(answer: str, context: str) -> CitationReport:
    response = with_retry(
        client.messages.parse,
        model=MODELS["standard"],
        max_tokens=1_200,
        system=CHECK_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Answer:\n{answer}\n\n"
                f"Documents:\n{context}"
            ),
        }],
        output_format=CitationReport,
        output_config={"effort": "low"},
    )
    log_usage(response, tool="assistant", stage="check")
    return response.parsed_output
