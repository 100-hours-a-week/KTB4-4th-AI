from datetime import datetime
from typing import Literal

from pydantic import Field

from app.api.schemas.common import ApiModel


class RecommendedItem(ApiModel):
    platform: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=255)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class RecommendationItems(ApiModel):
    items: list[RecommendedItem] = Field(max_length=20)


class FeedbackItem(ApiModel):
    platform: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=255)
    # LIKE·DISLIKE 는 상품 피드백, PURCHASED 는 본인이 추천 링크로 산 것,
    # RECEIVED 는 친구가 이 사용자의 선물 목록에서 골라 산 것이다.
    action: Literal["LIKE", "DISLIKE", "PURCHASED", "RECEIVED"]
    # 피드백을 남긴 목록. 아직 반영 방식은 같고 분석용으로 받는다.
    context: Literal["PERSONAL", "MY_GIFT", "FRIEND_GIFT"]
    occurred_at: datetime


class RecommendationFeedback(ApiModel):
    """지난 추천에 남긴 피드백. 최근 90일, 최신순 200개까지 보낸다."""

    items: list[FeedbackItem] = Field(max_length=200)


class RecommendationLists(ApiModel):
    generated_at: datetime
    self: RecommendationItems
    gift: RecommendationItems
