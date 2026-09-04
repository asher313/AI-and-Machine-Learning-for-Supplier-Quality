# src/sqm_ai/triage/rules.py
"""The deterministic first pass. Cheap, and always first."""
from __future__ import annotations
import re
from dataclasses import dataclass

I = re.IGNORECASE
SAFETY = re.compile(
    r"\b(crack|cracked|fracture|delamination|burn.?through"
    r"|embrittlement|foreign object debris|FOD)\b", I)
MATERIAL = re.compile(
    r"\b(wrong alloy|wrong material|mill cert|heat treat"
    r"|missing cert|certificate of conformance)\b", I)
COSMETIC = re.compile(
    r"\b(scratch|scuff|blemish|paint|handling mark)\b", I)

MUST = {"category", "severity", "supplier_id"}


@dataclass(frozen=True)
class RuleHit:
    """One field decided by one named rule."""
    field: str      # severity | category | supplier_id
    value: object
    rule: str       # the name that goes in the audit row


def apply_rules(
    text: str, part_number: str, part_map: dict[str, str]
) -> list[RuleHit]:
    """Every field a rule can decide, with the rule's name."""
    hits: list[RuleHit] = []
    safety = SAFETY.search(text)
    if safety:
        hits.append(RuleHit("severity", 5, "safety_keyword"))
    if MATERIAL.search(text):
        hits.append(RuleHit("category", "material", "mat_kw"))
    elif COSMETIC.search(text) and not safety:
        hits.append(RuleHit("category", "cosmetic", "cos_kw"))
    supplier = part_map.get(part_number.split("-")[0])
    if supplier is not None:
        hits.append(
            RuleHit("supplier_id", supplier, "part_family"))
    return hits


def decided(hits: list[RuleHit]) -> dict[str, object]:
    """Collapse hits to a field→value dict."""
    return {h.field: h.value for h in hits}


def is_complete(fields: dict[str, object]) -> bool:
    """True when no model call is needed at all."""
    return MUST <= set(fields)
