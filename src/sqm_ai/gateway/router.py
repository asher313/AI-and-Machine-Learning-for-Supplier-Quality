"""Explicit approved endpoints and transactionally reserved estimated spend."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

import psycopg

from sqm_ai.gateway.policy import BudgetExceeded, PolicyViolation


@dataclass(frozen=True)
class Endpoint:
    name: str
    model: str
    allowed_levels: frozenset[str]
    invoke: object
    quote: object  # Return a conservative Decimal token-charge estimate before dispatch.
    charge: object  # Compute actual supported token charges from returned usage.
    returned_models: frozenset[str] = frozenset()
    max_response_bytes: int = 262144

    def __post_init__(self):
        if (
            not self.name
            or not self.model
            or type(self.max_response_bytes) is not int
            or self.max_response_bytes < 1
        ):
            raise ValueError(
                "named model and positive response byte cap required"
            )
        object.__setattr__(
            self,
            "returned_models",
            frozenset(self.returned_models)
            or frozenset({self.model}),
        )

    def validate_response(self, response, request):
        """Generic text/usage contract before account-specific pricing is trusted.

        Pricing callbacks must additionally reject unsupported billable components.
        Returned aliases must be explicitly registered, never inferred from text.
        """
        if (
            not isinstance(response, dict)
            or response.get("model") not in self.returned_models
        ):
            raise PolicyViolation(["GW-RESPONSE-MODEL"])
        usage = response.get("usage")
        if (
            not isinstance(usage, dict)
            or not {"input_tokens", "output_tokens"}
            <= usage.keys()
        ):
            raise PolicyViolation(["GW-RESPONSE-USAGE"])

        def counters(value):
            for name, count in value.items():
                if name.endswith("_tokens"):
                    if type(count) is not int or count < 0:
                        raise PolicyViolation(
                            ["GW-RESPONSE-USAGE"]
                        )
                elif isinstance(count, dict):
                    counters(count)

        counters(usage)
        if usage["output_tokens"] > request["max_tokens"]:
            raise PolicyViolation(["GW-RESPONSE-USAGE"])
        text = response.get("text")
        if (
            isinstance(text, str)
            and len(text.encode("utf-8"))
            > self.max_response_bytes
        ):
            raise PolicyViolation(["GW-RESPONSE-SIZE"])


class ModelRouter:
    def __init__(self, endpoints):
        # Operator-provided (classification,tier)->Endpoint registry, not a user-provided URL.
        self.endpoints = dict(endpoints)

    def choose(self, enclave, tier):
        endpoint = self.endpoints.get((enclave, tier))
        if (
            endpoint is None
            or enclave not in endpoint.allowed_levels
        ):
            raise PolicyViolation(["GW-ROUTE"])
        return endpoint


def money(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("finite nonnegative amount required")
    return result.quantize(
        Decimal("0.00000001"), rounding=ROUND_CEILING
    )


class BudgetLedger:
    """UTC calendar-month estimates including concurrent in-flight reservations."""

    def __init__(self, dsn, cap, soft):
        self.dsn, self.cap, self.soft = (
            dsn,
            money(cap),
            Decimal(str(soft)),
        )
        if (
            self.cap <= 0
            or not self.soft.is_finite()
            or not 0 < self.soft <= 1
        ):
            raise ValueError(
                "positive cap and soft fraction in (0,1] required"
            )

    def reserve(self, trace_id, quote):
        quote = money(quote)
        with psycopg.connect(self.dsn) as conn:
            period = conn.execute(
                "SELECT date_trunc('month',now() AT TIME ZONE 'UTC')::date"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO sqm.gateway_budget_months(period,cap) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (period, self.cap),
            )
            spent, reserved, cap = conn.execute(
                "SELECT spent,reserved,cap FROM sqm.gateway_budget_months WHERE period=%s FOR UPDATE",
                (period,),
            ).fetchone()
            if cap != self.cap:
                raise ValueError(
                    "shared monthly cap differs; explicit operator migration required"
                )
            if spent + reserved + quote > cap:
                raise BudgetExceeded(
                    "estimated monthly reservation cap exceeded"
                )
            conn.execute(
                "INSERT INTO sqm.gateway_reservations VALUES (%s,%s,%s,NULL,'reserved')",
                (trace_id, period, quote),
            )
            conn.execute(
                "UPDATE sqm.gateway_budget_months SET reserved=reserved+%s WHERE period=%s",
                (quote, period),
            )
            return (
                "soft"
                if spent + reserved + quote
                >= self.soft * self.cap
                else "ok"
            )

    def settle(self, trace_id, actual=None, *, reconcile=False):
        """Reconcile uncertain charges only from operator-verified billing evidence."""
        with psycopg.connect(self.dsn) as conn:
            period = conn.execute(
                "SELECT period FROM sqm.gateway_reservations WHERE trace_id=%s",
                (trace_id,),
            ).fetchone()[0]
            conn.execute(
                "SELECT period FROM sqm.gateway_budget_months WHERE period=%s FOR UPDATE",
                (period,),
            )
            period, quoted, status, recorded = conn.execute(
                "SELECT period,quoted,status,actual FROM sqm.gateway_reservations WHERE trace_id=%s FOR UPDATE",
                (trace_id,),
            ).fetchone()
            if status == "settled":
                if (
                    actual is not None
                    and money(actual) == recorded
                ):
                    return (
                        recorded <= quoted
                    )  # Safe repeat after an ambiguous acknowledgement.
                raise ValueError(
                    "settled charge differs; explicit accounting correction required"
                )
            if actual is None:
                conn.execute(
                    "UPDATE sqm.gateway_reservations SET status='uncertain' WHERE trace_id=%s",
                    (trace_id,),
                )
                return False
            if status == "uncertain" and not reconcile:
                raise ValueError(
                    "uncertain charge requires explicit reconciliation"
                )
            actual = money(actual)
            conn.execute(
                "UPDATE sqm.gateway_budget_months SET reserved=reserved-%s,spent=spent+%s WHERE period=%s",
                (quoted, actual, period),
            )
            conn.execute(
                "UPDATE sqm.gateway_reservations SET status='settled',actual=%s WHERE trace_id=%s",
                (actual, trace_id),
            )
            return actual <= quoted
