import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from fastapi.testclient import TestClient

from app.application.recommendation_engine import RecommendationEngine
from app.core.config import Settings
from app.domain.profile.models import TasteField, Visibility
from app.domain.recommendation import (
    CatalogProduct,
    CatalogSearchResult,
    ProductKey,
    RecommendationSignal,
    VectorSpace,
)
from app.infrastructure.persistence import InMemorySessionStore
from app.main import create_app


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


def test_app_wires_recommendations_with_backend_join_key_and_reason() -> None:
    timestamp = datetime(2026, 9, 22, tzinfo=UTC).isoformat()
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        session_store=InMemorySessionStore(),
        embedder=FakeEmbedder(),
        catalog_repository=FakeCatalog(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/recommendations/jobs",
            headers={"Authorization": "Bearer test-token"},
            json={
                "userId": 10293,
                "sessionId": None,
                "profile": {
                    "schemaVersion": "3.0",
                    "userId": 10293,
                    "summary": None,
                    "interests": [
                        {
                            "value": "커피",
                            "confidence": 0.9,
                            "linkRole": "query",
                            "visibility": "friends",
                            "intentType": "both",
                            "deferralSignal": False,
                            "deferralReason": None,
                            "evidence": "커피를 좋아해요",
                            "taxonomyPath": None,
                            "firstSeenAt": timestamp,
                            "updatedAt": timestamp,
                        }
                    ],
                    "hobbies": [],
                    "preferences": [],
                    "lifestyle": [],
                    "wants": [],
                    "unaffordable": [],
                    "consumables": [],
                    "owned": [],
                    "dislikes": [],
                    "constraints": [],
                    "axes": [],
                },
            },
        )

    assert response.status_code == 200
    assert response.json()["self"]["items"] == [
        {
            "platform": "coupang",
            "externalId": "content",
            "reason": "커피에 대한 관심과 잘 맞는 상품이에요.",
        }
    ]
    assert response.json()["gift"]["items"] == [
        {
            "platform": "coupang",
            "externalId": "content",
            "reason": "커피에 관심 있는 분에게 선물하기 좋은 상품이에요.",
        }
    ]
