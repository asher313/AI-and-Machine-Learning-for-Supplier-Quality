# src/sqm_ai/triage/trace.py
from __future__ import annotations
import hashlib, time, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog
from sqm_ai.llm import estimate_cost

log = structlog.get_logger()


def _h(text: str) -> str:
    """First 16 hex chars of the SHA-256 of the text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LLMTrace:
    """One row per model call. Never holds prompt text."""
    trace_id: str = field(
        default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_now)
    model: str = ""
    stage: str = ""           # "fast" | "standard"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    input_hash: str = ""
    output_hash: str = ""
    metadata: dict = field(default_factory=dict)


def traced(fn, description: str, **metadata):
    """Run fn(); emit one LLMTrace whatever happens."""
    trace = LLMTrace(input_hash=_h(description),
                     stage=metadata.get("stage", ""),
                     metadata=metadata)
    start = time.perf_counter()
    try:
        result, response = fn()
        u = response.usage
        trace.model = response.model
        trace.input_tokens = u.input_tokens
        trace.output_tokens = u.output_tokens
        trace.cache_read = u.cache_read_input_tokens or 0
        trace.cost_usd = estimate_cost(response.model, u)
        trace.output_hash = _h(result.model_dump_json())
        return result, trace
    finally:
        trace.latency_ms = int(
            (time.perf_counter() - start) * 1000)
        log.info("llm_call", **trace.__dict__)
