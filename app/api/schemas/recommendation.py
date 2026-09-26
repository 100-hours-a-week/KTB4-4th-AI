from datetime import datetime

from pydantic import Field

from app.api.schemas.common import ApiModel


class RecommendedItem(ApiModel):
    platform: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=255)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class RecommendationItems(ApiModel):
    items: list[RecommendedItem] = Field(max_length=20)


class RecommendationLists(ApiModel):
    generated_at: datetime
    self: RecommendationItems
    gift: RecommendationItems
