from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from app.api.schemas.common import ApiModel

TasteField = Literal[
    "interests",
    "hobbies",
    "preferences",
    "lifestyle",
    "wants",
    "unaffordable",
    "consumables",
    "owned",
    "dislikes",
    "constraints",
]
LinkRole = Literal["query", "filter", "weight"]


class ProfileItem(ApiModel):
    value: str = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    link_role: LinkRole
    visibility: Literal["private", "friends", "public"]
    intent_type: Literal["need", "want", "both"] | None
    deferral_signal: bool
    deferral_reason: Literal["justification", "price", "timing"] | None = None
    evidence: str = Field(min_length=1, max_length=40)
    taxonomy_path: list[str] | None
    first_seen_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_deferral_reason(self) -> Self:
        if self.deferral_signal and self.deferral_reason is None:
            raise ValueError("deferralReason is required when deferralSignal is true")
        if not self.deferral_signal and self.deferral_reason is not None:
            raise ValueError("deferralReason must be null when deferralSignal is false")
        return self


class TasteProfile(ApiModel):
    schema_version: Literal["3.0"]
    user_id: str = Field(min_length=1)
    summary: str | None
    interests: list[ProfileItem]
    hobbies: list[ProfileItem]
    preferences: list[ProfileItem]
    lifestyle: list[ProfileItem]
    wants: list[ProfileItem]
    unaffordable: list[ProfileItem]
    consumables: list[ProfileItem]
    owned: list[ProfileItem]
    dislikes: list[ProfileItem]
    constraints: list[ProfileItem]
    axes: list[str] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_item_count(self) -> Self:
        item_count = sum(
            len(items)
            for items in (
                self.interests,
                self.hobbies,
                self.preferences,
                self.lifestyle,
                self.wants,
                self.unaffordable,
                self.consumables,
                self.owned,
                self.dislikes,
                self.constraints,
            )
        )
        if item_count > 12:
            raise ValueError("TasteProfile can contain at most 12 ProfileItems")
        return self


class ProfileKeywords(ApiModel):
    taste: list[str] = Field(max_length=3)
    interest: list[str] = Field(max_length=3)
