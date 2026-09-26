from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.domain.profile.models import DeferralReason, TasteField, Visibility
from app.domain.recommendation import (
    CatalogProduct,
    ProductKey,
    RecommendationMode,
    RecommendationSignal,
    VectorMatch,
    VectorSpace,
    build_search_queries,
    rank_recommendations,
)

NOW = datetime(2026, 9, 22, tzinfo=UTC)


def _signal(
    field: TasteField,
    value: str,
    *,
    confidence: float = 1.0,
    visibility: Visibility = Visibility.FRIENDS,
    age_days: int = 0,
    deferral_reason: DeferralReason | None = None,
) -> RecommendationSignal:
    return RecommendationSignal(
        field=field,
        value=value,
        confidence=confidence,
        visibility=visibility,
        updated_at=NOW - timedelta(days=age_days),
        deferral_reason=deferral_reason,
    )


def _product(platform: str, external_id: str, name: str) -> CatalogProduct:
    return CatalogProduct(
        key=ProductKey(platform=platform, external_id=external_id),
        name=name,
        price=Decimal("10000.00"),
    )


def test_v1_query_mapping_uses_only_positive_conversation_signals() -> None:
    queries = build_search_queries(
        [
            _signal(TasteField.INTERESTS, "커피", confidence=0.9),
            _signal(TasteField.HOBBIES, "캠핑", confidence=0.8),
            _signal(TasteField.WANTS, "휴대용 그라인더", confidence=0.7),
            _signal(
                TasteField.UNAFFORDABLE,
                "고급 그라인더",
                confidence=0.6,
                deferral_reason=DeferralReason.PRICE,
            ),
            _signal(TasteField.CONSUMABLES, "원두", confidence=0.5),
            _signal(TasteField.DISLIKES, "강한 향", confidence=1.0),
        ]
    )

    by_value = {query.signal.value: query for query in queries}
    assert "강한 향" not in by_value
    assert by_value["커피"].space == VectorSpace.CONTENT
    assert by_value["캠핑"].space == VectorSpace.USAGE
    assert by_value["휴대용 그라인더"].modes == frozenset(
        {RecommendationMode.SELF, RecommendationMode.GIFT}
    )
    assert by_value["고급 그라인더"].space == VectorSpace.GIFT
    assert by_value["고급 그라인더"].modes == frozenset({RecommendationMode.GIFT})
    assert by_value["원두"].modes == frozenset({RecommendationMode.SELF})


def test_multi_signal_bonus_can_promote_a_broadly_matching_product() -> None:
    coffee, camping = build_search_queries(
        [
            _signal(TasteField.INTERESTS, "커피"),
            _signal(TasteField.HOBBIES, "캠핑"),
        ]
    )
    broad = _product("coupang", "broad", "캠핑 머그컵")
    narrow = _product("coupang", "narrow", "커피 도구")

    ranked = rank_recommendations(
        [
            VectorMatch(query=coffee, product=broad, similarity=0.75),
            VectorMatch(query=camping, product=broad, similarity=0.75),
            VectorMatch(query=coffee, product=narrow, similarity=0.80),
        ],
        mode=RecommendationMode.SELF,
        now=NOW,
    )

    assert ranked[0].product == broad
    assert len(ranked[0].matches) == 2
    assert 0.0 <= ranked[0].score <= 1.0


def test_gift_deferral_boost_and_private_reason_are_applied() -> None:
    ordinary, deferred = build_search_queries(
        [
            _signal(TasteField.WANTS, "머그컵", confidence=0.8, visibility=Visibility.PRIVATE),
            _signal(
                TasteField.UNAFFORDABLE,
                "고급 그라인더",
                confidence=0.8,
                visibility=Visibility.PRIVATE,
                deferral_reason=DeferralReason.JUSTIFICATION,
            ),
        ]
    )
    ordinary_product = _product("coupang", "ordinary", "머그컵")
    deferred_product = _product("coupang", "deferred", "그라인더")

    ranked = rank_recommendations(
        [
            VectorMatch(query=ordinary, product=ordinary_product, similarity=0.8),
            VectorMatch(query=deferred, product=deferred_product, similarity=0.8),
        ],
        mode=RecommendationMode.GIFT,
        now=NOW,
    )

    assert ranked[0].product == deferred_product
    assert "고급 그라인더" not in ranked[0].reason
    assert ranked[0].reason == "평소 관심사와 잘 맞아 선물하기 좋은 상품이에요."


def test_recency_decay_prefers_a_recent_signal() -> None:
    recent, old = build_search_queries(
        [
            _signal(TasteField.INTERESTS, "최근 취향", age_days=0),
            _signal(TasteField.INTERESTS, "오래된 취향", age_days=28),
        ]
    )
    recent_product = _product("coupang", "recent", "최근 상품")
    old_product = _product("coupang", "old", "오래된 상품")

    ranked = rank_recommendations(
        [
            VectorMatch(query=recent, product=recent_product, similarity=0.6),
            VectorMatch(query=old, product=old_product, similarity=0.9),
        ],
        mode=RecommendationMode.SELF,
        now=NOW,
    )

    assert ranked[0].product == recent_product


def test_composite_product_key_deduplicates_only_within_the_same_platform() -> None:
    (query,) = build_search_queries([_signal(TasteField.INTERESTS, "커피")])
    coupang = _product("coupang", "123", "쿠팡 상품")
    other = _product("other", "123", "다른 플랫폼 상품")

    ranked = rank_recommendations(
        [
            VectorMatch(query=query, product=coupang, similarity=0.9),
            VectorMatch(query=query, product=coupang, similarity=0.8),
            VectorMatch(query=query, product=other, similarity=0.7),
        ],
        mode=RecommendationMode.SELF,
        now=NOW,
    )

    assert [item.product.key for item in ranked] == [coupang.key, other.key]
    assert len(ranked[0].matches) == 2


def test_results_are_capped_at_twenty_without_padding() -> None:
    (query,) = build_search_queries([_signal(TasteField.INTERESTS, "커피")])
    matches = [
        VectorMatch(
            query=query,
            product=_product("coupang", str(index), f"상품 {index}"),
            similarity=0.9 - index / 100,
        )
        for index in range(25)
    ]

    ranked = rank_recommendations(
        matches,
        mode=RecommendationMode.SELF,
        now=NOW,
    )
    sparse = rank_recommendations(
        matches[:3],
        mode=RecommendationMode.SELF,
        now=NOW,
    )

    assert len(ranked) == 20
    assert [item.rank for item in ranked] == list(range(20))
    assert len(sparse) == 3


def test_diversity_pick_does_not_break_descending_score_order() -> None:
    coffee_query, camping_query = build_search_queries(
        [
            _signal(TasteField.INTERESTS, "커피", confidence=1.0),
            _signal(TasteField.HOBBIES, "캠핑", confidence=0.5),
        ]
    )
    coffee_matches = [
        VectorMatch(
            query=coffee_query,
            product=_product("coupang", f"coffee-{index}", f"커피 상품 {index}"),
            similarity=0.9 - index / 100,
        )
        for index in range(5)
    ]
    camping_product = _product("coupang", "camping", "캠핑 의자")
    camping_match = VectorMatch(query=camping_query, product=camping_product, similarity=0.9)

    ranked = rank_recommendations(
        [*coffee_matches, camping_match],
        mode=RecommendationMode.SELF,
        now=NOW,
        limit=3,
    )

    scores = [item.score for item in ranked]
    assert camping_product in [item.product for item in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[-1].product == camping_product
    assert [item.rank for item in ranked] == [0, 1, 2]
