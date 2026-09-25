import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from app.application.recommendation_engine import RecommendationBatch
from app.application.recommendation_service import V1RecommendationService
from app.domain.recommendation import (
    CatalogProduct,
    ProductKey,
    RankedRecommendation,
    RecommendationMode,
)


class FakeEngine:
    async def recommend(self, signals, *, now=None, limit: int = 20) -> RecommendationBatch:
        assert [signal.value for signal in signals] == ["캠핑"]
        product = CatalogProduct(
            key=ProductKey(platform="coupang", external_id="12345"),
            name="캠핑 머그컵",
            price=Decimal("32000.00"),
        )
        self_item = RankedRecommendation(
            product=product,
            mode=RecommendationMode.SELF,
            rank=0,
            score=0.8,
            reason="캠핑에 대한 관심과 잘 맞는 상품이에요.",
            matches=(),
        )
        gift_item = RankedRecommendation(
            product=product,
            mode=RecommendationMode.GIFT,
            rank=0,
            score=0.7,
            reason="캠핑에 관심 있는 분에게 선물하기 좋은 상품이에요.",
            matches=(),
        )
        return RecommendationBatch(self=(self_item,), gift=(gift_item,))


def test_service_returns_composite_key_and_reason_for_both_lists() -> None:
    timestamp = datetime(2026, 9, 22, tzinfo=UTC).isoformat()
    service = V1RecommendationService(FakeEngine())

    result = asyncio.run(
        service.recommend_lists(
            {
                "profile": {
                    "interests": [],
                    "hobbies": [
                        {
                            "value": "캠핑",
                            "confidence": 0.9,
                            "visibility": "friends",
                            "updatedAt": timestamp,
                            "deferralReason": None,
                        }
                    ],
                    "wants": [],
                    "unaffordable": [],
                    "consumables": [],
                }
            }
        )
    )

    assert datetime.fromisoformat(result["generatedAt"])
    assert result["self"]["items"] == [
        {
            "platform": "coupang",
            "externalId": "12345",
            "score": 8.0,
            "reason": "캠핑에 대한 관심과 잘 맞는 상품이에요.",
        }
    ]
    assert result["gift"]["items"] == [
        {
            "platform": "coupang",
            "externalId": "12345",
            "score": 7.0,
            "reason": "캠핑에 관심 있는 분에게 선물하기 좋은 상품이에요.",
        }
    ]
