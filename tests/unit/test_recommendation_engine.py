import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from app.application.recommendation_engine import RecommendationEngine
from app.domain.profile.models import TasteField, Visibility
from app.domain.recommendation import (
    CatalogExclusion,
    CatalogProduct,
    CatalogSearchResult,
    ProductKey,
    RecommendationGuide,
    RecommendationSignal,
    VectorSpace,
)


class FakeEmbedder:
    space_id = "test-space"

    def __init__(self) -> None:
        self.query_batches: list[list[str]] = []

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("the online recommendation path must not embed passages")

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.query_batches.append(list(texts))
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeCatalog:
    def __init__(self) -> None:
        self.calls: list[tuple[VectorSpace, str, int]] = []
        self.filters: list[tuple[VectorSpace, Sequence[str] | None, bool]] = []

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
        self.calls.append((space, embedding_space_id, limit))
        self.filters.append((space, categories, gift_available_only))
        product = CatalogProduct(
            key=ProductKey(platform="coupang", external_id=space.value),
            name=f"{space.value} 상품",
            price=Decimal("10000.00"),
            gift_available=True,
        )
        return [CatalogSearchResult(product=product, similarity=0.9)]


def test_engine_batches_query_embeddings_and_searches_each_vector_space() -> None:
    embedder = FakeEmbedder()
    catalog = FakeCatalog()
    engine = RecommendationEngine(embedder=embedder, catalog=catalog)
    now = datetime(2026, 9, 22, tzinfo=UTC)

    result = asyncio.run(
        engine.recommend(
            [
                RecommendationSignal(
                    field=TasteField.INTERESTS,
                    value="커피",
                    confidence=0.9,
                    visibility=Visibility.FRIENDS,
                    updated_at=now,
                ),
                RecommendationSignal(
                    field=TasteField.HOBBIES,
                    value="캠핑",
                    confidence=0.8,
                    visibility=Visibility.FRIENDS,
                    updated_at=now,
                ),
                RecommendationSignal(
                    field=TasteField.UNAFFORDABLE,
                    value="고급 그라인더",
                    confidence=0.7,
                    visibility=Visibility.PRIVATE,
                    updated_at=now,
                ),
            ],
            now=now,
        )
    )

    # 관심사·취미마다 선물 전용 gift 쿼리가 하나씩 더 붙는다.
    assert len(embedder.query_batches) == 1
    assert len(embedder.query_batches[0]) == 5
    assert catalog.calls == [
        (VectorSpace.CONTENT, "test-space", 80),
        (VectorSpace.USAGE, "test-space", 80),
        (VectorSpace.GIFT, "test-space", 80),
        (VectorSpace.GIFT, "test-space", 80),
        (VectorSpace.GIFT, "test-space", 80),
    ]
    # 선물 전용 쿼리만 처음부터 선물 풀 안에서 찾는다.
    assert [gift_only for _, _, gift_only in catalog.filters] == [False, False, True, True, True]
    assert {item.product.key.external_id for item in result.self} == {"content", "usage"}
    assert {item.product.key.external_id for item in result.gift} == {
        "content",
        "usage",
        "gift",
    }


class GuideCatalog:
    """쿼리 벡터 [0.0]에는 두 상품을, 제외 벡터 [1.0]에는 향 강한 상품을 돌려준다."""

    def __init__(self) -> None:
        self.strong = CatalogProduct(
            key=ProductKey(platform="coupang", external_id="strong"),
            name="머스크 향수",
            price=Decimal("10000.00"),
            gift_available=True,
        )
        self.light = CatalogProduct(
            key=ProductKey(platform="coupang", external_id="light"),
            name="무향 바디로션",
            price=Decimal("10000.00"),
            gift_available=True,
        )
        self.named = CatalogProduct(
            key=ProductKey(platform="coupang", external_id="named"),
            name="강한 향 캔들",
            price=Decimal("10000.00"),
        )

    async def ping(self) -> bool:
        return True

    async def search(
        self,
        *,
        vector,
        space,
        embedding_space_id,
        limit,
        categories=None,
        gift_available_only=False,
        exclusion=None,
    ):
        # 선물 전용 gift 쿼리는 이 테스트의 관심사가 아니다.
        if space == VectorSpace.GIFT:
            return []
        if vector == [0.0]:
            return [
                CatalogSearchResult(product=self.strong, similarity=0.9),
                CatalogSearchResult(product=self.light, similarity=0.8),
                CatalogSearchResult(product=self.named, similarity=0.7),
            ]
        # 벡터 순서: 쿼리 [0.0], 선물 쿼리 [1.0], 제외 [2.0], 선호 [3.0]
        if vector == [2.0]:
            return [CatalogSearchResult(product=self.strong, similarity=0.8)]
        return [CatalogSearchResult(product=self.light, similarity=0.7)]


def test_engine_excludes_disliked_products_and_boosts_preferred_ones() -> None:
    embedder = FakeEmbedder()
    catalog = GuideCatalog()
    engine = RecommendationEngine(embedder=embedder, catalog=catalog)
    now = datetime(2026, 9, 22, tzinfo=UTC)

    result = asyncio.run(
        engine.recommend(
            [
                RecommendationSignal(
                    field=TasteField.INTERESTS,
                    value="향수",
                    confidence=0.9,
                    visibility=Visibility.FRIENDS,
                    updated_at=now,
                )
            ],
            now=now,
            guide=RecommendationGuide(exclusions=("강한 향",), preferences=("무향",)),
        )
    )

    assert embedder.query_batches == [
        ["향수 관련 제품", "향수에 관심 있는 사람이 받으면 좋은 선물", "강한 향", "무향"]
    ]
    assert [item.product.key.external_id for item in result.self] == ["light"]


class OnboardingCatalog:
    """검색마다 같은 후보를 돌려주고, 검색에 넘어온 제외 조건을 남긴다."""

    def __init__(self, products: Sequence[CatalogProduct]) -> None:
        self.products = products
        self.exclusions: list[tuple[VectorSpace, CatalogExclusion | None]] = []

    async def ping(self) -> bool:
        return True

    async def search(
        self,
        *,
        vector,
        space,
        embedding_space_id,
        limit,
        categories=None,
        gift_available_only=False,
        exclusion=None,
    ):
        self.exclusions.append((space, exclusion))
        return [CatalogSearchResult(product=product, similarity=0.9) for product in self.products]


def _product(external_id: str, name: str, category: str, subcategory: str | None, document=None):
    return CatalogProduct(
        key=ProductKey(platform="kakao", external_id=external_id),
        name=name,
        price=Decimal("10000.00"),
        category=category,
        subcategory=subcategory,
        gift_available=True,
        content_document=document,
    )


def test_engine_passes_onboarding_exclusion_to_search_and_drops_term_matches() -> None:
    from app.domain.profile.onboarding import Allergy, GiftExclusion
    from app.domain.recommendation import build_catalog_exclusion

    catalog = OnboardingCatalog(
        [
            _product("americano", "아이스 아메리카노 T 2잔", "voucher", "cafe_drink"),
            # 카페 교환권이지만 케이크에 넛츠가 들어 있다. content 문서에서 걸러야 한다.
            _product(
                "nut_cake",
                "아메리카노 T + 얼 그레이 케이크",
                "voucher",
                "cafe_drink",
                "교환권 카페 음료. 카라멜 넛츠 케이크와 아메리카노 세트다.",
            ),
            _product("wine_glass", "크리스탈 와인잔 2P 세트", "living", "kitchen"),
            _product("mug", "커피 머그컵", "living", "kitchen"),
        ]
    )
    engine = RecommendationEngine(embedder=FakeEmbedder(), catalog=catalog)
    now = datetime(2026, 10, 15, tzinfo=UTC)
    exclusion = build_catalog_exclusion(
        allergies=[Allergy.PEANUT, Allergy.NUTS], gift_exclusions=[GiftExclusion.ALCOHOL]
    )

    result = asyncio.run(
        engine.recommend(
            [
                RecommendationSignal(
                    field=TasteField.INTERESTS,
                    value="커피",
                    confidence=0.9,
                    visibility=Visibility.FRIENDS,
                    updated_at=now,
                )
            ],
            now=now,
            guide=RecommendationGuide(catalog_exclusion=exclusion),
        )
    )

    # 대분류·세부 카테고리 필터는 쿼리 검색마다 SQL로 넘어간다.
    assert catalog.exclusions
    assert all(passed is exclusion for _, passed in catalog.exclusions)
    assert {item.product.key.external_id for item in result.self} == {"americano", "mug"}
    assert {item.product.key.external_id for item in result.gift} == {"americano", "mug"}
