from datetime import datetime
from typing import Literal

from pydantic import Field

from app.api.schemas.common import ApiModel, ConversationRoomId, UserId
from app.api.schemas.profile import TasteProfile

RecommendationMode = Literal["self", "gift"]


class PriceRange(ApiModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)
    currency: Literal["KRW"]


class RecommendationFunnel(ApiModel):
    retrieved: int = Field(ge=0)
    after_hard_filter: int = Field(ge=0)
    after_score_floor: int = Field(ge=0)
    returned: int = Field(ge=0)


class RecommendedItem(ApiModel):
    platform: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1)


class RecommendationResult(ApiModel):
    recommendation_id: str
    generated_at: datetime
    mode: RecommendationMode
    price_range: PriceRange | None
    items: list[RecommendedItem] = Field(max_length=20)
    empty_reason: (
        Literal[
            "NO_PROFILE_SIGNAL",
            "ALL_EXCLUDED",
            "NO_RELEVANT_CANDIDATE",
            "CATALOG_GAP",
        ]
        | None
    )
    suggestion: str | None = None
    funnel: RecommendationFunnel


class CreateRecommendationListsRequest(ApiModel):
    user_id: UserId
    session_id: ConversationRoomId | None
    profile: TasteProfile


class RecommendationLists(ApiModel):
    self: RecommendationResult
    gift: RecommendationResult


class CreateRecommendationRequest(ApiModel):
    user_id: UserId
    mode: RecommendationMode
    profile: TasteProfile
    limit: int = Field(default=20, ge=1, le=20)
