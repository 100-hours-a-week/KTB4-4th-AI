from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from app.domain.recommendation import (
    CatalogProduct,
    CatalogSearchResult,
    ProductKey,
    VectorSpace,
)

if TYPE_CHECKING:
    import asyncpg

# pgvector 의 <=> 는 코사인 '거리'(가까울수록 작음)를 돌려준다.
# ranking.py 는 코사인 '유사도'(가까울수록 큼)를 기대하므로 1 - distance 로 뒤집는다.
# 이걸 빠뜨리면 가장 안 맞는 상품이 1등으로 올라오는데 에러는 나지 않는다.
#
# ORDER BY 는 뒤집지 않은 거리 그대로 쓴다. 나중에 HNSW 인덱스를 걸었을 때
# 연산자 표현식이 그대로여야 인덱스를 탄다.
_SEARCH_SQL = """
SELECT p.platform,
       p.external_id,
       p.name,
       p.price,
       1 - (d.embedding <=> $1::vector) AS similarity
  FROM catalog_product_documents AS d
  JOIN catalog_products AS p
    ON p.platform = d.platform
   AND p.external_id = d.external_id
 WHERE d.space = $2
   AND d.embedding_model_version = $3
   AND p.is_active
   AND p.processing_status = 'ready'
 ORDER BY d.embedding <=> $1::vector
 LIMIT $4
"""


class PostgresCatalogRepository:
    """추천 검색 읽기 전용 어댑터. ai_catalog_api 계정으로 붙는다."""

    def __init__(self, *, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ping(self) -> bool:
        async with self._pool.acquire() as connection:
            return await connection.fetchval("SELECT 1") == 1

    async def search(
        self,
        *,
        vector: Sequence[float],
        space: VectorSpace,
        embedding_space_id: str,
        limit: int,
    ) -> Sequence[CatalogSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                _SEARCH_SQL,
                _to_vector_literal(vector),
                space.value,
                embedding_space_id,
                limit,
            )
        return [_to_result(row) for row in rows]


def _to_result(row: Any) -> CatalogSearchResult:
    price = row["price"]
    return CatalogSearchResult(
        product=CatalogProduct(
            key=ProductKey(platform=row["platform"], external_id=row["external_id"]),
            name=row["name"],
            price=None if price is None else Decimal(str(price)),
        ),
        similarity=float(row["similarity"]),
    )


def _to_vector_literal(vector: Sequence[float]) -> str:
    """pgvector 는 '[1,2,3]' 형태의 텍스트를 vector 로 캐스팅한다."""
    if not vector:
        raise ValueError("query vector must not be empty")
    return json.dumps([float(value) for value in vector])
