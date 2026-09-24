from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Any

import aiomysql

from app.domain.catalog.sync import SourceProduct
from app.domain.recommendation.models import ProductKey

# 상품 수가 늘어도 뒤쪽 페이지가 느려지지 않도록 OFFSET 대신 키셋 페이징을 쓴다.
_PAGE_SQL = """
SELECT platform_type AS platform, external_id, name, price
  FROM {table}
 WHERE (platform_type, external_id) > (%s, %s)
 ORDER BY platform_type, external_id
 LIMIT %s
"""


class MySQLProductSource:
    """백엔드 MySQL에서 상품을 읽기만 한다. 쓰기 경로는 두지 않는다."""

    def __init__(
        self,
        *,
        pool: aiomysql.Pool,
        table: str = "products",
    ) -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError("table name must be alphanumeric")
        self._pool = pool
        self._sql = _PAGE_SQL.format(table=table)

    async def ping(self) -> bool:
        async with self._pool.acquire() as connection, connection.cursor() as cursor:
            await cursor.execute("SELECT 1")
            return await cursor.fetchone() is not None

    async def iter_products(
        self,
        *,
        batch_size: int,
    ) -> AsyncIterator[Sequence[SourceProduct]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cursor_platform = ""
        cursor_external_id = ""
        while True:
            async with self._pool.acquire() as connection:
                async with connection.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        self._sql,
                        (cursor_platform, cursor_external_id, batch_size),
                    )
                    rows = await cursor.fetchall()
            if not rows:
                return
            yield [_to_product(row) for row in rows]
            cursor_platform = str(rows[-1]["platform"])
            cursor_external_id = str(rows[-1]["external_id"])
            if len(rows) < batch_size:
                return


def _to_product(row: dict[str, Any]) -> SourceProduct:
    price = row.get("price")
    return SourceProduct(
        key=ProductKey(
            platform=str(row["platform"]),
            external_id=str(row["external_id"]),
        ),
        name=str(row["name"]),
        price=None if price is None else Decimal(str(price)),
    )
