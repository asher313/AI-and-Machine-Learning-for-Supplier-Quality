"""Per-call metadata; hashes support comparison, not content recovery or secrecy."""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import structlog

from sqm_ai.llm import estimate_cost

log = structlog.get_logger()


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class LLMTrace:
    trace_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    model: str = ""
    stage: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read: int | None = None
    cache_written: int | None = None
    latency_ms: int = 0
    cost_usd: float | None = None
    request_hash: str = ""
    output_hash: str = ""
    request_id: str | None = None
    status: str = "started"
    error_type: str | None = None


def traced(fn, request, *, stage, trace=None):
    """One trace per attempted HTTP request; retries call this again."""
    trace = trace if trace is not None else LLMTrace()
    trace.model, trace.stage = request["model"], stage
    trace.request_hash = _h(
        json.dumps(request, sort_keys=True, separators=(",", ":"))
    )
    start = time.perf_counter()
    try:
        response = fn()
        u = response.usage
        trace.model = response.model
        trace.request_id = getattr(response, "_request_id", None)
        trace.input_tokens, trace.output_tokens = (
            u.input_tokens,
            u.output_tokens,
        )
        trace.cache_read = u.cache_read_input_tokens or 0
        trace.cache_written = u.cache_creation_input_tokens or 0
        trace.cost_usd = estimate_cost(response.model, u)
        blocks = [
            b.model_dump(exclude_none=True)
            for b in response.content
        ]
        trace.output_hash = _h(json.dumps(blocks, sort_keys=True))
        trace.status = response.stop_reason
        return response, trace
    except Exception as exc:
        trace.status, trace.error_type = (
            "error",
            type(exc).__name__,
        )
        raise
    finally:
        trace.latency_ms = int(
            (time.perf_counter() - start) * 1000
        )
        log.info("llm_attempt", **asdict(trace))
