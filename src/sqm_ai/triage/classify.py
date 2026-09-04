# src/sqm_ai/triage/classify.py
from sqm_ai.errors import ModelInferenceError
from sqm_ai.llm import MODELS, client
from sqm_ai.triage.schema import TriageResult

TAXONOMY = open("prompts/ncr_taxonomy.md").read()  # ~3,000 tok
SYSTEM = (
    "You are an aerospace supplier-quality engineer. "
    "Classify the nonconformance report using the taxonomy "
    "provided. Severity 5 is safety-critical and stops a "
    "line; 1 is trivial. Confidence is your honest "
    "probability that BOTH category and severity are right. "
    "If the text is too vague to classify, say so in your "
    "reasoning and give a confidence below 0.5. Use only a "
    "supplier id from the shortlist you are given.")

_schema = TriageResult.model_json_schema()
_schema["additionalProperties"] = False   # strict needs it
TRIAGE_TOOL = {
    "name": "record_triage",
    "description": "Record the triage of one NCR.",
    "strict": True,
    "input_schema": _schema,
}


def call_model(model: str, ncr, neighbours: list[str]):
    """One classification call. Returns (result, response)."""
    response = client.messages.create(
        model=model,
        max_tokens=700,
        system=[
            {"type": "text", "text": SYSTEM},
            {"type": "text", "text": TAXONOMY,
             "cache_control": {"type": "ephemeral"}},
        ],
        tools=[TRIAGE_TOOL],
        tool_choice={"type": "tool", "name": "record_triage"},
        messages=[{"role": "user",
                   "content": render_ncr(ncr, neighbours)}],
        output_config={"effort": "low"},
    )
    if response.stop_reason != "tool_use":
        raise ModelInferenceError(response.stop_reason)
    block = next(
        b for b in response.content if b.type == "tool_use")
    return TriageResult.model_validate(block.input), response


def classify(ncr, nbrs) -> tuple[TriageResult, str]:
    """Fast tier first; escalate only when it is unsure."""
    result, resp = call_model(MODELS["fast"], ncr, nbrs)
    if result.confidence >= 0.85:
        return result, resp.model
    result, resp = call_model(MODELS["standard"], ncr, nbrs)
    return result, resp.model
