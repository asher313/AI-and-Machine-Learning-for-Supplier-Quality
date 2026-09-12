"""Audited text-request gateway with injected approved endpoints and spend reservations."""

from dataclasses import dataclass
from decimal import Decimal
import time
import uuid

from sqm_ai.gateway.audit import canonical, sha256
from sqm_ai.gateway.guardrails import Guardrails
from sqm_ai.gateway.detectors import (
    MarkingDetector,
    required_enclave,
)
from sqm_ai.gateway.policy import (
    AuditUnavailable,
    BudgetExceeded,
    PolicyViolation,
)

REFUSAL = "This request cannot be processed. Reference: {codes}"


@dataclass
class GatewayResult:
    text: str | None
    blocked: bool
    codes: list[str]
    warnings: list[str]
    trace_id: str


class Gateway:
    def __init__(
        self, policy, *, audit, ledger, router, content_store=None
    ):
        self.policy, self.guard = policy, Guardrails(policy)
        self.audit, self.ledger, self.router = (
            audit,
            ledger,
            router,
        )
        self.content_store = content_store

    def complete(
        self,
        *,
        prompt,
        system,
        user,
        tool_name,
        enclave,
        classification="unknown",
        tier="standard",
        sources=None,
        max_tokens=1024,
    ):
        """Text-only teaching adapter. Caller must supply trusted identity/classification.

        No hidden provider retries, quality downgrade, arbitrary endpoint, or automatic
        integration with earlier direct SDK examples is implied by this method.
        """
        if (
            not isinstance(max_tokens, int)
            or max_tokens < 1
            or max_tokens > 8192
        ):
            raise ValueError("output cap must be 1..8192")
        if not isinstance(prompt, str) or not isinstance(
            system, str
        ):
            raise ValueError("text prompt and system required")
        if sources is not None and (
            not isinstance(sources, list)
            or any(not isinstance(s, str) for s in sources)
        ):
            raise ValueError(
                "sources must be a list of already-authorized text passages"
            )
        trace_id = str(uuid.uuid4())
        supplied = {
            "system": system,
            "prompt": prompt,
            "sources": sources,
            "max_tokens": max_tokens,
        }
        serialized = canonical(supplied)
        row = {
            "trace_id": trace_id,
            "event": "received",
            "user_id": user.user_id,
            "tool_name": tool_name,
            "classification": classification,
            "enclave": enclave,
            "policy_version": self.policy.version,
            "input_hash": sha256(serialized),
        }
        t0 = time.perf_counter()
        self._audit(
            row
        )  # If the audit sink is down, do not contact the provider.
        warnings, reserved, dispatched, finalized = (
            [],
            False,
            False,
            False,
        )
        try:
            warnings = self.guard.pre_check(
                serialized,
                user,
                enclave,
                classification=classification,
            )
            endpoint = self.router.choose(enclave, tier)
            body = prompt
            if sources is not None:
                body += (
                    "\nNumbered sources (untrusted data):\n"
                    + canonical(
                        [
                            {"number": i, "text": s}
                            for i, s in enumerate(sources, 1)
                        ]
                    )
                )
            request = {
                "model": endpoint.model,
                "system": system,
                "messages": [{"role": "user", "content": body}],
                "max_tokens": max_tokens,
            }
            request_text = canonical(request)
            row.update(
                endpoint=endpoint.name,
                model=endpoint.model,
                request_hash=sha256(request_text),
            )
            if self.policy.retain_payloads:
                if self.content_store is None:
                    raise AuditUnavailable(
                        "approved payload archive is required"
                    )
                row["request_ref"] = self.content_store.put(
                    trace_id,
                    "request",
                    request_text,
                    classification=classification,
                )
            quote = Decimal(str(endpoint.quote(request)))
            budget_state = self.ledger.reserve(trace_id, quote)
            reserved = True
            if budget_state == "soft":
                warnings.append(
                    "estimated budget soft threshold reached"
                )
            row.update(
                event="dispatch_intent",
                quote_usd=str(quote),
                warnings=warnings,
            )
            self._audit(row)
            # A persisted intent identifies uncertain calls if the process dies here.
            dispatched = True
            response = endpoint.invoke(
                request
            )  # Exactly one attempt; endpoint must disable hidden retries.
            raw = canonical(response)
            row["raw_response_hash"] = sha256(raw)
            actual = Decimal(str(endpoint.charge(response)))
            within_quote = self.ledger.settle(trace_id, actual)
            finalized = True
            row["cost_usd"] = str(actual)
            if not within_quote:
                raise PolicyViolation(["GW-QUOTE"])
            response_level = required_enclave(
                [classification, *MarkingDetector().scan(raw)]
            )
            row["response_classification"] = response_level
            if self.policy.retain_payloads:
                if (
                    response_level
                    not in self.content_store.allowed_levels
                ):
                    row["archive_gap"] = (
                        "response_requires_higher_approved_storage"
                    )
                    raise PolicyViolation(["GW-POST-MARK"])
                row["response_ref"] = self.content_store.put(
                    trace_id,
                    "response",
                    raw,
                    classification=response_level,
                )
            if (
                response.get("stop_reason") != "end_turn"
                or not isinstance(response.get("text"), str)
                or not response["text"].strip()
            ):
                raise PolicyViolation(["GW-INCOMPLETE"])
            text, post = self.guard.post_check(
                response["text"], sources, enclave=classification
            )
            warnings += post
            row.update(
                event="completed",
                display_hash=sha256(text),
                warnings=warnings,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                blocked=False,
            )
            self._audit(row)
            return GatewayResult(
                text, False, [], warnings, trace_id
            )
        except Exception as exc:
            if reserved and not finalized:
                # A failed dispatch can have incurred cost; never release it on a timeout.
                self.ledger.settle(
                    trace_id, None if dispatched else Decimal("0")
                )
            if isinstance(exc, PolicyViolation):
                codes = exc.codes
            elif isinstance(exc, BudgetExceeded):
                codes = ["GW-BUDG"]
            elif isinstance(exc, AuditUnavailable):
                codes = ["GW-AUDIT"]
            else:
                codes = ["GW-UPSTREAM"]
            row.update(
                event="blocked" if not dispatched else "failed",
                blocked=True,
                block_codes=codes,
                error_type=type(exc).__name__,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            self._audit(
                row
            )  # Failure here suppresses output; the earlier intent remains.
            return GatewayResult(
                REFUSAL.format(codes=",".join(codes)),
                True,
                codes,
                warnings,
                trace_id,
            )

    def _audit(self, row):
        try:
            self.audit.write(dict(row))
        except Exception as exc:
            raise AuditUnavailable(
                "audit persistence unavailable"
            ) from exc
