import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from app.application.recommendation_engine import RecommendationEngine
from app.application.recommendation_service import recommendation_feedback
from app.domain.profile.models import TasteField, Visibility
from app.domain.recommendation import (
    CatalogExclusion,
    CatalogProduct,
    CatalogSearchResult,
    FeedbackAction,
    FeedbackSummary,
    ProductFeedback,
    ProductKey,
    RecommendationSignal,
    VectorSpace,
)
from app.domain.recommendation.feedback import (
    MAX_SIGNAL_WEIGHT,
    MIN_SIGNAL_WEIGHT,
    OWNED_SIMILAR_WEIGHT,
)

NOW = datetime(2026, 11, 2, tzinfo=UTC)


def key(external_id: str) -> ProductKey:
    return ProductKey(platform="coupang", external_id=external_id)


def product(external_id: str, subcategory: str | None = None) -> CatalogProduct:
    return CatalogProduct(
        key=key(external_id),
        name=f"상품 {external_id}",
        price=Decimal("10000"),
        gift_available=True,
        subcategory=subcategory,
    )


def feedback(**actions: FeedbackAction) -> FeedbackSummary:
    return FeedbackSummary.of(
        ProductFeedback(key=key(external_id), action=action)
        for external_id, action in actions.items()
    )


def test_one_action_per_product_with_dislike_and_purchase_first() -> None:
    summary = FeedbackSummary.of(
        [
            ProductFeedback(key("1"), FeedbackAction.LIKE),
            ProductFeedback(key("1"), FeedbackAction.DISLIKE),
            ProductFeedback(key("2"), FeedbackAction.LIKE),
            ProductFeedback(key("2"), FeedbackAction.PURCHASED),
            ProductFeedback(key("3"), FeedbackAction.LIKE),
        ]
    )

    assert summary.actions == {
        key("1"): FeedbackAction.DISLIKE,
        key("2"): FeedbackAction.PURCHASED,
        key("3"): FeedbackAction.LIKE,
    }
    assert summary.excluded == frozenset({key("1"), key("2")})


def test_feedback_payload_is_parsed_from_backend_shape() -> None:
    summary = recommendation_feedback(
        {
            "feedback": {
                "items": [
                    {
                        "platform": "COUPANG",
                        "externalId": "9",
                        "action": "RECEIVED",
                        "context": "FRIEND_GIFT",
                        "occurredAt": "2026-10-30T00:00:00Z",
                    }
                ]
            }
        }
    )

    assert summary is not None
    assert summary.actions == {key("9"): FeedbackAction.RECEIVED}
    assert recommendation_feedback({}) is None


class FeedbackCatalog:
    """커피 쿼리는 1·2·3번을, 캠핑 쿼리는 4·5번을 찾는다. 2번과 5번은 같은 세부 카테고리다."""

    def __init__(self) -> None:
        self.calls = 0

    async def ping(self) -> bool:
        return True

    async def search(
        self,
        *,
        vector: Sequence[float],
        space: VectorSpace,
        embedding_space_id: str,
        limit: int,
        categories: Sequence[str] | None = None,
        gift_available_only: bool = False,
        exclusion: CatalogExclusion | None = None,
    ) -> Sequence[CatalogSearchResult]:
        if vector[0] == 0.0:
            return [
                CatalogSearchResult(product("1"), 0.8),
                CatalogSearchResult(product("2", "drip"), 0.8),
                CatalogSearchResult(product("3"), 0.8),
            ]
        return [
            CatalogSearchResult(product("4"), 0.8),
            CatalogSearchResult(product("5", "drip"), 0.8),
        ]


class TwoQueryEmbedder:
    space_id = "test-space"

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0 if "커피" in text else 1.0] for text in texts]


def _signals() -> list[RecommendationSignal]:
    return [
        RecommendationSignal(
            field=TasteField.INTERESTS,
            value=value,
            confidence=0.8,
            visibility=Visibility.FRIENDS,
            updated_at=NOW,
        )
        for value in ("커피", "캠핑")
    ]


def _scores(summary: FeedbackSummary | None) -> dict[str, float]:
    engine = RecommendationEngine(embedder=TwoQueryEmbedder(), catalog=FeedbackCatalog())
    batch = asyncio.run(engine.recommend(_signals(), now=NOW, feedback=summary))
    return {item.product.key.external_id: item.score for item in batch.self}


def test_disliked_and_owned_products_are_excluded_and_their_tastes_reweighted() -> None:
    baseline = _scores(None)
    scores = _scores(feedback(**{"1": FeedbackAction.DISLIKE, "4": FeedbackAction.LIKE}))

    assert "1" not in scores
    # 싫어요를 받은 상품을 고른 커피 취향은 낮아지고, 좋아요를 받은 캠핑 취향은 올라간다.
    assert scores["3"] < baseline["3"]
    assert scores["5"] > baseline["5"]


def test_products_like_a_purchased_one_are_lowered() -> None:
    baseline = _scores(None)
    scores = _scores(feedback(**{"2": FeedbackAction.PURCHASED}))

    assert "2" not in scores
    # 같은 세부 카테고리(drip)의 5번은 낮추고, 커피 취향은 산 만큼 맞았으니 조금 올린다.
    assert scores["5"] < baseline["5"]
    assert scores["3"] > baseline["3"]


def test_feedback_weights_stay_in_a_narrow_band() -> None:
    from app.domain.recommendation.feedback import owned_similar_weights, signal_weights
    from app.domain.recommendation.models import SearchQuery, VectorMatch

    query = SearchQuery(
        query_id="q1",
        signal=_signals()[0],
        modes=frozenset(),
        space=VectorSpace.CONTENT,
        text="커피",
    )
    many = [VectorMatch(query, product(str(index), "drip"), 0.8) for index in range(20)]

    disliked = feedback(**{str(index): FeedbackAction.DISLIKE for index in range(19)})
    liked = feedback(**{str(index): FeedbackAction.LIKE for index in range(19)})
    bought = feedback(**{"0": FeedbackAction.PURCHASED})

    assert signal_weights(many, disliked) == {"q1": MIN_SIGNAL_WEIGHT}
    assert signal_weights(many, liked) == {"q1": MAX_SIGNAL_WEIGHT}
    weights = owned_similar_weights(many, bought)
    assert key("0") not in weights
    assert set(weights.values()) == {OWNED_SIMILAR_WEIGHT}
