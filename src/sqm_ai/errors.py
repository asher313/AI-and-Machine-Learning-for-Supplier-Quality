# src/sqm_ai/errors.py
class SQMError(Exception):
    """Base class: catch this to catch anything sqm_ai raises."""


class ModelInferenceError(SQMError):
    """A model call failed after retries."""


class SchemaError(SQMError):
    """Data did not match the shape the code expected."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")
