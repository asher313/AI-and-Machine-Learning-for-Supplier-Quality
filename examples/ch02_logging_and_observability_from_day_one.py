# Chapter 2 — 2.5 Logging and Observability From Day One
import structlog

log = structlog.get_logger()

log.info(
    "ncr_classified",
    ncr_id="NCR-2026-0042",
    supplier_id="S-0417",
    category="dimensional",
    confidence=0.92,
    latency_ms=340,
    model="haiku",        # the MODELS key, not a raw id
)
