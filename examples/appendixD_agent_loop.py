# Appendix D — D.4 The No-Framework Agent Loop (Chapter 19.3)
# Reprint for memorization. The shipped file is
# src/sqm_ai/agent/loop.py, and Chapter 19.3's version is
# what it holds: this reprint imports
# `from sqm_ai.log import get_logger`, which EXAMPLE_CANON
# ruling R7 rules out (there is no sqm_ai.log.get_logger;
# the logger is structlog.get_logger()).
# src/sqm_ai/agent/loop.py
from sqm_ai.agent.tools import IMPLS, TOOLS
from sqm_ai.llm import MODELS, client, log_usage, with_retry
from sqm_ai.log import get_logger

log = get_logger(__name__)

SYSTEM = (
    "You are a supplier-quality engineer processing one "
    "nonconformance report. Use the tools in a sensible "
    "order: classify it, look up the supplier's history, "
    "then state a disposition in plain sentences. Draft a "
    "corrective action request only if severity is 3 or "
    "higher. When you are finished, reply with the "
    "disposition and one sentence of justification."
)


class AgentStalled(RuntimeError):
    """The loop hit its step cap without finishing."""


def run_agent(
    ncr_id: str, supplier_id: str, description: str,
    max_steps: int = 8,
) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"NCR: {ncr_id}\nSupplier: {supplier_id}\n"
            f"Description: {description}"
        ),
    }]

    for step in range(max_steps):
        response = with_retry(
            client.messages.create,
            model=MODELS["standard"],
            max_tokens=1_024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        log_usage(response, agent="ncr", step=step)
        messages.append(
            {"role": "assistant", "content": response.content}
        )

        if response.stop_reason != "tool_use":
            return "".join(
                b.text for b in response.content
                if b.type == "text"
            )

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                out = IMPLS[block.name](**block.input)
                failed = False
            except Exception as exc:            # noqa: BLE001
                out = f"Error: {exc}"
                failed = True
            log.info(
                "agent_tool", step=step, tool=block.name,
                failed=failed,
            )
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(out),
                "is_error": failed,
            })
        messages.append({"role": "user", "content": results})

    raise AgentStalled(
        f"{ncr_id}: no answer in {max_steps} steps"
    )
