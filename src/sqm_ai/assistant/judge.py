# src/sqm_ai/assistant/judge.py
from pydantic import BaseModel, Field

from sqm_ai.llm import MODELS, client, log_usage, with_retry

JUDGE_SYSTEM = """\
You grade a quality-systems assistant against a reference
answer written by a senior engineer. Grade only these:
- correct: does the answer agree with the reference on every
  substantive point? A missing point is not a disagreement.
- complete: does it cover the reference's main points?
- cited: is every claim marked with a [N]?
Score 0.0 to 1.0. Do not reward length or fluency.
"""


class Verdict(BaseModel):
    correct: bool
    complete: bool
    cited: bool
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=20, max_length=400)


def judge(
    question: str, answer: str, reference: str
) -> Verdict:
    response = with_retry(
        client.messages.parse,
        model=MODELS["frontier"],
        max_tokens=600,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Assistant answer:\n{answer}\n\n"
                f"Reference answer:\n{reference}"
            ),
        }],
        output_format=Verdict,
    )
    log_usage(response, tool="assistant", stage="judge")
    return response.parsed_output
