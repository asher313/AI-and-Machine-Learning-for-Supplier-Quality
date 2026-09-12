"""Fictional policy labels require trusted identity and classification inputs."""

from dataclasses import dataclass, field
from decimal import Decimal
import re

ENCLAVE_RANK = {"open": 0, "internal": 1, "controlled": 2}


@dataclass(frozen=True)
class GatewayUser:
    user_id: str
    permissions: frozenset[str] = frozenset()
    allowed_levels: frozenset[str] = frozenset(
        {"open", "internal"}
    )


@dataclass(frozen=True)
class Policy:
    version: str = "teaching-v2"
    forbidden_patterns: tuple[str, ...] = ()
    validate_citations: bool = False
    injection_threshold: float = 0.8
    monthly_budget_usd: Decimal = Decimal("1500")
    soft_alert_fraction: Decimal = Decimal("0.80")
    retain_payloads: bool = True

    def __post_init__(self):
        for name in ("monthly_budget_usd", "soft_alert_fraction"):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite():
                raise ValueError("finite policy amounts required")
            object.__setattr__(self, name, value)
        if not self.version or self.monthly_budget_usd <= 0:
            raise ValueError(
                "policy version and positive cap required"
            )
        if (
            not 0 < self.soft_alert_fraction <= 1
            or not 0 < self.injection_threshold <= 1
        ):
            raise ValueError("policy fractions must be in (0,1]")
        for pattern in self.forbidden_patterns:
            re.compile(pattern)


class PolicyViolation(Exception):
    def __init__(self, codes, reasons=()):
        self.codes, self.reasons = list(codes), list(reasons)
        super().__init__(", ".join(codes))


class BudgetExceeded(Exception):
    pass


class AuditUnavailable(RuntimeError):
    pass
