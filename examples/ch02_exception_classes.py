# Chapter 2 — Exception classes
import structlog

from sqm_ai.errors import ModelInferenceError

log = structlog.get_logger()

try:
    result = model.predict(features)
except MemoryError:
    result = model.predict(features[:1000])   # degrade
except Exception as e:
    log.exception("inference_failed")
    raise ModelInferenceError(f"predict failed: {e}") from e
