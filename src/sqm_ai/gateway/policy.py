# sqm_ai/gateway/policy.py
from dataclasses import dataclass, field

ENCLAVE_RANK = {"open": 0, "internal": 1, "controlled": 2}


@dataclass(frozen=True)
class GatewayUser:
    user_id: str
    permissions: frozenset[str] = frozenset()


@dataclass
class Policy:
    forbidden_patterns: list[str] = field(default_factory=list)
    validate_citations: bool = False
    injection_threshold: float = 0.8
    monthly_budget_usd: float = 1500.0   # canon: Ch 1
    soft_alert_fraction: float = 0.80    # alert at $1,200


class PolicyViolation(Exception):
    """Raised by the pre-filter. Codes are user-facing."""

    def __init__(self, codes: list[str], reasons: list[str]):
        self.codes = codes
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class BudgetExceeded(Exception):
    pass
