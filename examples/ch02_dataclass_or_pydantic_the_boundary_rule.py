# Chapter 2 — Dataclass or Pydantic: the boundary rule
from dataclasses import dataclass


@dataclass(frozen=True)
class Classification:
    category: str
    severity: int
    confidence: float
