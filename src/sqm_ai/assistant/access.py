# src/sqm_ai/assistant/access.py
from dataclasses import dataclass

SOURCES = ("as9100", "manual", "car", "audit")


@dataclass(frozen=True)
class AssistantUser:
    user_id: str
    programs: tuple[str, ...] = ()   # ("NL-KESTREL",)
    sources: tuple[str, ...] = SOURCES

    @property
    def allowed_programs(self) -> list[str]:
        return sorted(self.programs)

    @property
    def allowed_source_types(self) -> list[str]:
        return list(self.sources)
