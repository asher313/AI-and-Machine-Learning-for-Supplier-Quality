# Chapter 20 — 20.4 The Root-Cause Agent
# src/sqm_ai/car/root_cause.py
from pydantic import BaseModel, Field

from sqm_ai.llm import MODELS, client, log_usage, with_retry


class Why(BaseModel):
    question: str = Field(min_length=8)
    answer: str = Field(min_length=8)
    evidence_ref: str = Field(min_length=3)


class Hypothesis(BaseModel):
    observation: str = Field(min_length=10)
    whys: list[Why] = Field(min_length=5, max_length=5)
    terminal_root_cause: str = Field(min_length=15)
    confidence: float = Field(ge=0.0, le=1.0)
    investigation_gaps: list[str]


class RootCauseSet(BaseModel):
    hypotheses: list[Hypothesis] = Field(
        min_length=3, max_length=3
    )


# Chapter 20 — 20.4 The Root-Cause Agent (continued)
ROOT_CAUSE_SYSTEM = """\
You are an aerospace root-cause analyst. You are given one
nonconformance, the supplier's recent record, and similar
past corrective actions. Propose exactly three distinct
hypotheses for the cause, using the five-why method.

Rules:
1. The three hypotheses must be genuinely different
   mechanisms, not three phrasings of one idea.
2. Each hypothesis has exactly five whys. Each why has a
   question, an answer, and evidence_ref: the id of the
   NCR, CAR, or audit finding that supports the answer.
3. If no supplied evidence supports a why, do not invent
   one. Write the answer as "UNKNOWN" with evidence_ref
   "none", and add the specific thing that would need to
   be checked to investigation_gaps.
4. A root cause must be a condition that can be removed.
   A restatement of the defect is not a root cause, and
   neither is a general statement about training,
   attention, or care unless the evidence names a
   specific procedure or record.
5. confidence is your honest probability that this
   hypothesis is the dominant cause. Three hypotheses
   with confidence 0.9 is a wrong answer.
6. Prefer a mechanism that explains the supplier's whole
   pattern over one that explains only this event, and
   say which you have done.
"""


def analyze(evidence: dict) -> RootCauseSet:
    response = with_retry(
        client.messages.parse,
        model=MODELS["frontier"],
        max_tokens=4_000,
        system=ROOT_CAUSE_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Nonconformance:\n{evidence['ncr_text']}\n\n"
                f"Supplier record (last 5 NCRs and CARs):\n"
                f"{evidence['history_text']}\n\n"
                f"Similar past corrective actions:\n"
                f"{evidence['similar_text']}"
            ),
        }],
        output_format=RootCauseSet,
    )
    log_usage(response, crew="car", role="root_cause")
    return response.parsed_output
