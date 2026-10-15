from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import pytest

from app.domain.catalog.subcategories import SUBCATEGORY_MIN_CONFIDENCE
from app.domain.recommendation import CatalogExclusion, VectorSpace
from app.infrastructure.persistence import postgres_catalog_repository as module
from app.infrastructure.persistence.postgres_catalog_repository import PostgresCatalogRepository


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.sql: str | None = None
        self.args: tuple[Any, ...] = ()

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.sql = sql
        self.args = args
        return self._rows

    async def fetchval(self, sql: str, *args: Any) -> int:
        self.sql = sql
        return 1


class FakePool:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.connection = FakeConnection(rows or [])

    @asynccontextmanager
    async def acquire(self):  # noqa: ANN201
        yield self.connection


def _row(external_id: str, similarity: float) -> dict[str, Any]:
    return {
        "platform": "coupang",
        "external_id": external_id,
        "name": f"캠핑 용품 {external_id}",
        "price": Decimal(30000),
        "category": "sports",
        "gift_available": True,
        "subcategory": "camping",
        "content_document": "레저·스포츠 캠핑 용품.",
        "similarity": similarity,
    }


def test_search_maps_rows_to_results() -> None:
    pool = FakePool([_row("1", 0.82), _row("2", 0.41)])
    repository = PostgresCatalogRepository(pool=pool)

    results = asyncio.run(
        repository.search(
            vector=[0.1, 0.2, 0.3],
            space=VectorSpace.USAGE,
            embedding_space_id="solar-embedding-2",
            limit=80,
        )
    )

    assert [result.product.key.external_id for result in results] == ["1", "2"]
    assert results[0].similarity == pytest.approx(0.82)
    assert results[0].product.price == Decimal(30000)
    assert results[0].product.category == "sports"
    assert results[0].product.gift_available is True
    assert results[0].product.subcategory == "camping"
    assert results[0].product.content_document == "레저·스포츠 캠핑 용품."


def test_search_passes_vector_as_pgvector_literal() -> None:
    pool = FakePool([])
    repository = PostgresCatalogRepository(pool=pool)

    asyncio.run(
        repository.search(
            vector=[0.1, 0.2],
            space=VectorSpace.GIFT,
            embedding_space_id="solar-embedding-2",
            limit=10,
        )
    )

    (
        vector_arg,
        space_arg,
        model_arg,
        limit_arg,
        categories_arg,
        gift_only_arg,
        excluded_categories_arg,
        excluded_subcategories_arg,
        min_confidence_arg,
    ) = pool.connection.args
    assert json.loads(vector_arg) == [0.1, 0.2]
    assert space_arg == "gift"
    assert model_arg == "solar-embedding-2"
    assert limit_arg == 10
    assert categories_arg is None
    assert gift_only_arg is False
    # 제외 조건이 없으면 빈 배열이라 아무것도 빼지 않는다.
    assert excluded_categories_arg == []
    assert excluded_subcategories_arg == []
    assert min_confidence_arg == SUBCATEGORY_MIN_CONFIDENCE


def test_search_passes_category_and_gift_available_filters() -> None:
    pool = FakePool([])
    repository = PostgresCatalogRepository(pool=pool)

    asyncio.run(
        repository.search(
            vector=[0.1, 0.2],
            space=VectorSpace.CONTENT,
            embedding_space_id="solar-embedding-2",
            limit=10,
            categories=("beauty", "living"),
            gift_available_only=True,
        )
    )

    categories_arg, gift_only_arg = pool.connection.args[4:6]
    assert categories_arg == ["beauty", "living"]
    assert gift_only_arg is True
    assert "p.category = ANY($5::text[])" in module._SEARCH_SQL
    assert "(NOT $6::boolean OR p.gift_available)" in module._SEARCH_SQL


def test_search_passes_onboarding_exclusion_as_structural_filters() -> None:
    pool = FakePool([])
    repository = PostgresCatalogRepository(pool=pool)

    asyncio.run(
        repository.search(
            vector=[0.1, 0.2],
            space=VectorSpace.CONTENT,
            embedding_space_id="solar-embedding-2",
            limit=10,
            exclusion=CatalogExclusion(
                categories=frozenset({"voucher", "food"}),
                subcategories=frozenset({"health:", "beauty:fragrance"}),
                name_terms=("와인",),
            ),
        )
    )

    excluded_categories_arg, excluded_subcategories_arg = pool.connection.args[6:8]
    assert excluded_categories_arg == ["food", "voucher"]
    assert excluded_subcategories_arg == ["beauty:fragrance", "health:"]
    assert "NOT (COALESCE(p.category, '') = ANY($7::text[]))" in module._SEARCH_SQL
    # 확신이 낮은 세부 카테고리는 빈 값으로 보고 "health:" 같은 키에 걸리게 한다.
    assert "WHEN p.subcategory_confidence >= $9 THEN p.subcategory" in module._SEARCH_SQL
    assert "|| ':' || COALESCE(s.subcategory, '') = ANY($8::text[])" in module._SEARCH_SQL


def test_search_converts_distance_to_similarity() -> None:
    """ranking.py 는 유사도를 기대한다. 거리를 그대로 넘기면 순위가 통째로 뒤집힌다."""
    assert "1 - (d.embedding <=> $1::vector) AS similarity" in module._SEARCH_SQL
    # 정렬은 나중에 HNSW 인덱스를 타도록 뒤집지 않은 거리 표현식 그대로 둔다.
    assert "ORDER BY d.embedding <=> $1::vector" in module._SEARCH_SQL


def test_search_only_returns_ready_and_active_products() -> None:
    assert "p.is_active" in module._SEARCH_SQL
    assert "p.processing_status = 'ready'" in module._SEARCH_SQL


def test_search_rejects_non_positive_limit() -> None:
    repository = PostgresCatalogRepository(pool=FakePool([]))

    with pytest.raises(ValueError):
        asyncio.run(
            repository.search(
                vector=[0.1],
                space=VectorSpace.CONTENT,
                embedding_space_id="solar-embedding-2",
                limit=0,
            )
        )


def test_search_rejects_empty_vector() -> None:
    repository = PostgresCatalogRepository(pool=FakePool([]))

    with pytest.raises(ValueError):
        asyncio.run(
            repository.search(
                vector=[],
                space=VectorSpace.CONTENT,
                embedding_space_id="solar-embedding-2",
                limit=10,
            )
        )


def test_engine_ranks_recommendations_through_the_repository() -> None:
    """임베딩 -> pgvector 조회 -> 랭킹까지 실제 경로로 한 번 돌려본다."""
    from datetime import UTC, datetime

    from app.application.recommendation_engine import RecommendationEngine
    from app.domain.profile.models import TasteField, Visibility
    from app.domain.recommendation import RecommendationSignal

    class FakeEmbedder:
        @property
        def space_id(self) -> str:
            return "solar-embedding-2"

        async def embed_passages(self, texts):  # noqa: ANN001, ANN201
            raise NotImplementedError

        async def embed_queries(self, texts):  # noqa: ANN001, ANN201
            return [[0.1, 0.2] for _ in texts]

    pool = FakePool([_row("1", 0.9), _row("2", 0.2)])
    engine = RecommendationEngine(
        embedder=FakeEmbedder(),
        catalog=PostgresCatalogRepository(pool=pool),
    )
    signal = RecommendationSignal(
        field=TasteField.HOBBIES,
        value="캠핑",
        confidence=0.9,
        visibility=Visibility.FRIENDS,
        updated_at=datetime.now(UTC),
    )

    batch = asyncio.run(engine.recommend([signal]))

    assert [item.product.key.external_id for item in batch.self] == ["1", "2"]
    assert batch.self[0].score > batch.self[1].score
    assert batch.gift
    # hobbies 는 usage 공간으로 찾고, 선물 목록용 gift 공간 쿼리가 하나 더 붙는다.
    # FakeConnection 은 마지막 호출 인자만 남기므로 마지막 검색이 gift 다.
    assert pool.connection.args[1] == "gift"
    assert pool.connection.args[5] is True
