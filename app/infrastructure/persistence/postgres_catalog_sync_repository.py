from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.domain.catalog.sync import SourceProduct, SyncMode

if TYPE_CHECKING:
    import asyncpg

_START_RUN_SQL = """
INSERT INTO catalog_sync_runs (mode) VALUES ($1) RETURNING run_id
"""

# 마지막 WHERE 절이 핵심이다. 값이 그대로인 행은 아예 쓰지 않아서
# 매 실행마다 전체 테이블을 다시 쓰고 오토바큠을 깨우는 일을 막는다.
#
# 카테고리는 원본 product_category 에 있는 것만 여기서 쓴다. 문서 생성 모델이 고른
# 카테고리(inferred)는 문서 입력이 그대로인 동안 유지한다. 가격만 바뀐 동기화가
# 모델이 고른 카테고리를 지우면 상품이 ready 인 채로 카테고리 탭에서 사라진다.
_UPSERT_SQL = """
INSERT INTO catalog_products (
    platform, external_id, name, price,
    source_hash, doc_input_hash,
    source_labels, category, category_source, themes, gift_available,
    is_active, processing_status, last_seen_run_id, source_synced_at
)
SELECT
    src.platform, src.external_id, src.name, src.price,
    src.source_hash, src.doc_input_hash,
    string_to_array(src.source_labels, chr(31)), src.category, src.category_source,
    string_to_array(src.themes, ','), src.gift_available,
    true, 'pending', $1, now()
FROM unnest(
        $2::text[], $3::text[], $4::text[], $5::numeric[], $6::text[], $7::text[],
        $8::text[], $9::text[], $10::text[], $11::text[], $12::boolean[]
     ) AS src(
        platform, external_id, name, price, source_hash, doc_input_hash,
        source_labels, category, category_source, themes, gift_available
     )
ON CONFLICT (platform, external_id) DO UPDATE SET
    name             = EXCLUDED.name,
    price            = EXCLUDED.price,
    source_hash      = EXCLUDED.source_hash,
    doc_input_hash   = EXCLUDED.doc_input_hash,
    source_labels    = EXCLUDED.source_labels,
    themes           = EXCLUDED.themes,
    category = CASE
        WHEN EXCLUDED.category IS NULL
             AND catalog_products.category_source = 'inferred'
             AND catalog_products.doc_input_hash = EXCLUDED.doc_input_hash
        THEN catalog_products.category
        ELSE EXCLUDED.category
    END,
    category_source = CASE
        WHEN EXCLUDED.category IS NULL
             AND catalog_products.category_source = 'inferred'
             AND catalog_products.doc_input_hash = EXCLUDED.doc_input_hash
        THEN catalog_products.category_source
        ELSE EXCLUDED.category_source
    END,
    category_confidence = CASE
        WHEN EXCLUDED.category IS NULL
             AND catalog_products.category_source = 'inferred'
             AND catalog_products.doc_input_hash = EXCLUDED.doc_input_hash
        THEN catalog_products.category_confidence
        ELSE NULL
    END,
    gift_available     = EXCLUDED.gift_available,
    is_active        = true,
    last_seen_run_id = EXCLUDED.last_seen_run_id,
    source_synced_at = now(),
    updated_at       = now(),
    processing_status = CASE
        WHEN catalog_products.doc_input_hash IS DISTINCT FROM EXCLUDED.doc_input_hash
        THEN 'pending'
        ELSE catalog_products.processing_status
    END
WHERE catalog_products.source_hash IS DISTINCT FROM EXCLUDED.source_hash
"""

# 이번 실행에서 한 번도 안 보인 상품. 행을 지우지 않고 내리기만 해서
# 품절됐다 돌아오는 상품의 문서와 벡터를 재사용한다.
_DEACTIVATE_SQL = """
UPDATE catalog_products
   SET is_active = false, updated_at = now()
 WHERE is_active
   AND (last_seen_run_id IS NULL OR last_seen_run_id <> $1)
"""

_TOUCH_SEEN_SQL = """
UPDATE catalog_products AS p
   SET last_seen_run_id = $1
  FROM unnest($2::text[], $3::text[]) AS src(platform, external_id)
 WHERE p.platform = src.platform
   AND p.external_id = src.external_id
   AND p.last_seen_run_id IS DISTINCT FROM $1
"""

_FINISH_RUN_SQL = """
UPDATE catalog_sync_runs
   SET status = $2,
       received_count = $3,
       changed_count = $4,
       deactivated_count = $5,
       finished_at = now()
 WHERE run_id = $1
"""


class PostgresCatalogSyncRepository:
    def __init__(self, *, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def start_run(self, *, mode: SyncMode) -> int:
        async with self._pool.acquire() as connection:
            return int(await connection.fetchval(_START_RUN_SQL, mode.value))

    async def upsert_products(
        self,
        products: Sequence[SourceProduct],
        *,
        run_id: int,
    ) -> int:
        if not products:
            return 0
        platforms = [product.key.platform for product in products]
        external_ids = [product.key.external_id for product in products]
        async with self._pool.acquire() as connection, connection.transaction():
            status = await connection.execute(
                _UPSERT_SQL,
                run_id,
                platforms,
                external_ids,
                [product.name for product in products],
                [product.price for product in products],
                [product.source_hash for product in products],
                [product.doc_input_hash for product in products],
                # asyncpg 는 text[][] 를 unnest 로 행마다 풀 수 없어서 구분자로 이어 보낸다.
                [_LABEL_SEPARATOR.join(product.labels) for product in products],
                [_category_value(product) for product in products],
                [_category_source_value(product) for product in products],
                [",".join(theme.value for theme in product.themes) for product in products],
                [product.gift_available for product in products],
            )
            # 값이 그대로여서 위 upsert가 건너뛴 행도 이번 실행에서 본 것은 맞다.
            # 표시를 안 남기면 deactivate_missing이 멀쩡한 상품을 내린다.
            await connection.execute(_TOUCH_SEEN_SQL, run_id, platforms, external_ids)
        return _affected_rows(status)

    async def deactivate_missing(self, *, run_id: int) -> int:
        async with self._pool.acquire() as connection:
            return _affected_rows(await connection.execute(_DEACTIVATE_SQL, run_id))

    async def finish_run(
        self,
        *,
        run_id: int,
        received_count: int,
        changed_count: int,
        deactivated_count: int,
        failed: bool = False,
    ) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute(
                _FINISH_RUN_SQL,
                run_id,
                "failed" if failed else "succeeded",
                received_count,
                changed_count,
                deactivated_count,
            )


# 라벨 사이 구분자. 상품명이나 라벨에 나올 일이 없는 ASCII 단위 구분자(chr 31)를 쓴다.
_LABEL_SEPARATOR = "\x1f"


def _category_value(product: SourceProduct) -> str | None:
    category, _ = product.category_assignment
    return category.value if category is not None else None


def _category_source_value(product: SourceProduct) -> str | None:
    _, source = product.category_assignment
    return source.value if source is not None else None


def _affected_rows(status: str) -> int:
    """asyncpg는 'UPDATE 12' 같은 명령 태그를 돌려준다."""
    parts = status.split()
    return int(parts[-1]) if parts and parts[-1].isdigit() else 0
