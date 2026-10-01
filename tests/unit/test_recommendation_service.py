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
    async def recommend(
        self, signals, *, now=None, limit: int = 20, guide=None
    ) -> RecommendationBatch:
        assert [signal.value for signal in signals] == ["캠핑"]
        assert guide.exclusions == ("강한 향",)
        assert guide.preferences == ("가벼운 것",)
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
                    "dislikes": [{"value": "강한 향"}],
                    "preferences": [{"value": "가벼운 것"}],
                }
            }
        )
    )

    assert datetime.fromisoformat(result["generatedAt"])
    assert result["self"]["items"] == [
        {
            "platform": "coupang",
            "externalId": "12345",
            "score": 0.8,
            "reason": "캠핑에 대한 관심과 잘 맞는 상품이에요.",
        }
    ]
    assert result["gift"]["items"] == [
        {
            "platform": "coupang",
            "externalId": "12345",
            "score": 0.7,
            "reason": "캠핑에 관심 있는 분에게 선물하기 좋은 상품이에요.",
        }
    ]


def test_recommendation_inputs_route_preferences_by_aspect() -> None:
    from app.application.recommendation_service import recommendation_inputs

    timestamp = datetime(2026, 9, 22, tzinfo=UTC).isoformat()

    def preference(value: str, aspect: str | None) -> dict[str, object]:
        return {
            "value": value,
            "confidence": 0.8,
            "rankScore": 0.8,
            "visibility": "friends",
            "updatedAt": timestamp,
            "aspect": aspect,
        }

    signals, guide = recommendation_inputs(
        {
            "profile": {
                "preferences": [
                    preference("조용한 시골 여행", "situation"),
                    preference("고소한 커피", "sensory"),
                    preference("머리 비우는 시간", "motive"),
                    preference("가벼운 것", None),
                ],
                "dislikes": [{"value": "강한 향"}],
            }
        }
    )

    # 취향은 모두 쿼리 재료로 넘어가고, 측면에 따라 어떤 쿼리가 될지는 쿼리 빌더가 정한다.
    assert {
        (signal.value, signal.aspect.value if signal.aspect else None) for signal in signals
    } == {
        ("조용한 시골 여행", "situation"),
        ("고소한 커피", "sensory"),
        ("머리 비우는 시간", "motive"),
        ("가벼운 것", None),
    }
    assert guide.preferences == ("고소한 커피", "가벼운 것")
    assert guide.exclusions == ("강한 향",)
