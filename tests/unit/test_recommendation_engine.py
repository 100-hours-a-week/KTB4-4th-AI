import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from app.application.recommendation_engine import RecommendationEngine
from app.domain.profile.models import TasteField, Visibility
from app.domain.recommendation import (
    CatalogProduct,
    CatalogSearchResult,
    ProductKey,
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

    async def ping(self) -> bool:
        return True

    async def search(
        self,
        *,
        vector: Sequence[float],
        space: VectorSpace,
        embedding_space_id: str,
        limit: int,
    ) -> Sequence[CatalogSearchResult]:
        self.calls.append((space, embedding_space_id, limit))
        product = CatalogProduct(
            key=ProductKey(platform="coupang", external_id=space.value),
            name=f"{space.value} 상품",
            price=Decimal("10000.00"),
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

    assert len(embedder.query_batches) == 1
    assert len(embedder.query_batches[0]) == 3
    assert catalog.calls == [
        (VectorSpace.CONTENT, "test-space", 80),
        (VectorSpace.USAGE, "test-space", 80),
        (VectorSpace.GIFT, "test-space", 80),
    ]
    assert {item.product.key.external_id for item in result.self} == {"content", "usage"}
    assert {item.product.key.external_id for item in result.gift} == {
        "content",
        "usage",
        "gift",
    }
