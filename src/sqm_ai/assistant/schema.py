"""Structured claims make citation coverage explicit before rendering prose."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    citations: list[int] = Field(min_length=1, max_length=8)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refused: bool
    claims: list[Claim] = Field(max_length=12)

    @model_validator(mode="after")
    def coherent(self):
        if self.refused == bool(self.claims):
            raise ValueError(
                "refusal has no claims; an answer must have claims"
            )
        if sum(len(c.text.split()) for c in self.claims) > 250:
            raise ValueError("answer exceeds 250 words")
        return self
