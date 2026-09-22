from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from app.api.schemas.common import ApiModel, UserId

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
    user_id: UserId
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
    def validate_active_working_set(self) -> Self:
        query_count = sum(
            len(items)
            for items in (
                self.interests,
                self.hobbies,
                self.wants,
                self.unaffordable,
                self.consumables,
            )
        )
        weight_count = len(self.preferences) + len(self.lifestyle)
        if query_count > 6:
            raise ValueError("TasteProfile can contain at most 6 active query items")
        if weight_count > 3:
            raise ValueError("TasteProfile can contain at most 3 active weight items")
        if len(self.owned) > 3:
            raise ValueError("TasteProfile can contain at most 3 active owned items")
        return self


class ProfileKeywords(ApiModel):
    taste: list[str] = Field(max_length=3)
    interest: list[str] = Field(max_length=3)
