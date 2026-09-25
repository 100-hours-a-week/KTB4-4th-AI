from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING

from app.application.catalog_document_service import ProductEnrichment
from app.domain.catalog.sync import SourceProduct
from app.domain.recommendation.models import ProductKey

if TYPE_CHECKING:
    import asyncpg

# FOR UPDATE SKIP LOCKED 로 다른 워커가 이미 집어간 행을 건너뛴다.
# 배치를 여러 개 띄워도 같은 상품을 두 번 처리하지 않는다.
_CLAIM_SQL = """
WITH picked AS (
    SELECT platform, external_id
      FROM catalog_products
     WHERE processing_status = 'pending'
       AND is_active
     ORDER BY updated_at
     LIMIT $1
     FOR UPDATE SKIP LOCKED
)
UPDATE catalog_products AS p
   SET processing_status = 'processing',
       processing_error = NULL,
       updated_at = now()
  FROM picked
 WHERE p.platform = picked.platform
   AND p.external_id = picked.external_id
RETURNING p.platform, p.external_id, p.name, p.price
"""

_UPSERT_DOCUMENT_SQL = """
INSERT INTO catalog_product_documents (
    platform, external_id, space,
    document_text, document_hash, document_version,
    embedding_model_version, embedding, generated_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, now())
ON CONFLICT (platform, external_id, space, embedding_model_version) DO UPDATE SET
    document_text  = EXCLUDED.document_text,
    document_hash  = EXCLUDED.document_hash,
    document_version = EXCLUDED.document_version,
    embedding      = EXCLUDED.embedding,
    generated_at   = now()
"""

_MARK_READY_SQL = """
UPDATE catalog_products
   SET processing_status = 'ready',
       processing_error = NULL,
       updated_at = now()
 WHERE platform = $1 AND external_id = $2
"""

_MARK_FAILED_SQL = """
UPDATE catalog_products
   SET processing_status = 'failed',
       processing_error = left($3, 500),
       updated_at = now()
 WHERE platform = $1 AND external_id = $2
"""

_REQUEUE_FAILED_SQL = """
UPDATE catalog_products
   SET processing_status = 'pending',
       processing_error = NULL,
       updated_at = now()
 WHERE processing_status = 'failed'
   AND is_active
"""

# 배치가 죽으면 processing 인 채로 남아 영영 다시 안 잡힌다.
# 오래된 processing 을 주기적으로 pending 으로 되돌린다.
_RELEASE_STALE_SQL = """
UPDATE catalog_products
   SET processing_status = 'pending',
       updated_at = now()
 WHERE processing_status = 'processing'
   AND updated_at < now() - make_interval(secs => $1)
"""


class PostgresCatalogDocumentRepository:
    def __init__(self, *, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def claim_pending(self, *, limit: int) -> Sequence[SourceProduct]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch(_CLAIM_SQL, limit)
        return [
            SourceProduct(
                key=ProductKey(platform=row["platform"], external_id=row["external_id"]),
                name=row["name"],
                price=None if row["price"] is None else Decimal(str(row["price"])),
            )
            for row in rows
        ]

    async def save_documents(
        self,
        enrichment: ProductEnrichment,
        *,
        vectors: Sequence[Sequence[float]],
        embedding_model_version: str,
    ) -> None:
        documents = enrichment.documents
        if len(vectors) != len(documents.documents):
            raise ValueError("vector count does not match document count")
        key = enrichment.key
        # 문서 3개와 상태 전환을 한 트랜잭션에 묶는다. content 만 저장되고
        # gift 가 빠진 채 ready 가 되면 추천이 조용히 반쪽만 돌아간다.
        async with self._pool.acquire() as connection, connection.transaction():
            for document, vector in zip(documents.documents, vectors, strict=True):
                await connection.execute(
                    _UPSERT_DOCUMENT_SQL,
                    key.platform,
                    key.external_id,
                    document.space.value,
                    document.text,
                    document.text_hash,
                    documents.document_version,
                    embedding_model_version,
                    _to_vector_literal(vector),
                )
            await connection.execute(_MARK_READY_SQL, key.platform, key.external_id)

    async def mark_failed(self, key: ProductKey, *, reason: str) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(_MARK_FAILED_SQL, key.platform, key.external_id, reason)

    async def requeue_failed(self) -> int:
        async with self._pool.acquire() as connection:
            status = await connection.execute(_REQUEUE_FAILED_SQL)
        parts = status.split()
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0

    async def release_stale(self, *, older_than_seconds: int) -> int:
        async with self._pool.acquire() as connection:
            status = await connection.execute(_RELEASE_STALE_SQL, float(older_than_seconds))
        parts = status.split()
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0


def _to_vector_literal(vector: Sequence[float]) -> str:
    """pgvector 는 '[1,2,3]' 형태의 텍스트를 vector 로 캐스팅한다."""
    return json.dumps([float(value) for value in vector])
