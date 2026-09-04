# src/sqm_ai/log.py
import logging
import sys

import structlog

# Free text and drawings may hold controlled data. Never log them.
CONTROLLED_KEYS = {"defect_description", "drawing", "photo_path"}


def redact_controlled(logger, method, event_dict):
    for key in CONTROLLED_KEYS & event_dict.keys():
        event_dict[key] = "[redacted]"
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_controlled,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
