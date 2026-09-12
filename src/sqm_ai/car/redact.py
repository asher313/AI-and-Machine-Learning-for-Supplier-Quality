"""Per-run deterministic pseudonyms; no claim of anonymization or clearance."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Redactor:
    forward: dict[str, str]
    backward: dict[str, str]

    @classmethod
    def build(cls, names: dict[str, str], aliases=None):
        fwd, back, owners = {}, {}, {}
        aliases = aliases or {}
        for index, (sid, name) in enumerate(
            sorted(names.items()), 1
        ):
            token = f"[[SUPPLIER_{index:04d}]]"
            back[token] = name
            for alias in [sid, name, *aliases.get(sid, [])]:
                if not alias.strip() or "[[SUPPLIER_" in alias:
                    raise ValueError(
                        "invalid supplier identity or alias"
                    )
                folded = alias.casefold()
                if folded in owners and owners[folded] != token:
                    raise ValueError("ambiguous identity alias")
                owners[folded] = token
                fwd[alias] = token
        return cls(fwd, back)

    def _pattern(self):
        # Boundary-aware single pass avoids substring collisions and recursive replacement.
        alternatives = "|".join(
            re.escape(s)
            for s in sorted(self.forward, key=len, reverse=True)
        )
        return (
            re.compile(
                r"(?<!\w)(?:" + alternatives + r")(?!\w)",
                re.IGNORECASE,
            )
            if alternatives
            else None
        )

    def out(self, text):
        if re.search(r"\[\[SUPPLIER_\d+\]\]", text):
            raise ValueError(
                "raw source contains a reserved pseudonym token"
            )
        pattern = self._pattern()
        lookup = {
            k.casefold(): v for k, v in self.forward.items()
        }
        return (
            pattern.sub(
                lambda m: lookup[m.group().casefold()], text
            )
            if pattern
            else text
        )

    def back(self, text):
        pattern = re.compile(r"\[\[SUPPLIER_\d+\]\]")

        def replace(match):
            token = match.group()
            if token not in self.backward:
                raise ValueError(
                    "unknown pseudonym in model output"
                )
            return self.backward[token]

        return pattern.sub(replace, text)

    def leaked(self, text):
        pattern = self._pattern()
        return (
            list(
                dict.fromkeys(
                    m.group() for m in pattern.finditer(text)
                )
            )
            if pattern
            else []
        )

    def outbound(self, value):
        """Transform string VALUES only; source IDs and keys require explicit schema design."""
        if isinstance(value, str):
            result = self.out(value)
            if self.leaked(result):
                raise ValueError(
                    "known identity survived redaction"
                )
            return result
        if isinstance(value, list):
            return [self.outbound(v) for v in value]
        if isinstance(value, dict):
            return {k: self.outbound(v) for k, v in value.items()}
        return value
