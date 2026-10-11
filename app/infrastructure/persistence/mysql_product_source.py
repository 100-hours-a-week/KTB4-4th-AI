from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Any

import aiomysql

from app.domain.catalog.sync import SourceProduct
from app.domain.recommendation.models import ProductKey

# 상품 수가 늘어도 뒤쪽 페이지가 느려지지 않도록 OFFSET 대신 키셋 페이징을 쓴다.
_PAGE_SQL = """
SELECT platform_type AS platform, external_id, name, price{category}{themes}
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
        category_column: str | None = None,
        themes_column: str | None = None,
    ) -> None:
        """category_column 은 상품 카테고리(예: category), themes_column 은 선물 테마 목록
        (예: gift_themes) 컬럼이다.

        카테고리를 비워 두면 문서 생성 모델이 고른다. 테마가 없으면 교환권·명품·패션만
        선물 목록 후보가 된다.
        """
        for identifier in (table, category_column or "x", themes_column or "x"):
            if not identifier.replace("_", "").isalnum():
                raise ValueError("table and column names must be alphanumeric")
        self._pool = pool
        self._sql = _PAGE_SQL.format(
            table=table,
            category=f", {category_column} AS product_category" if category_column else "",
            themes=f", {themes_column} AS gift_themes" if themes_column else "",
        )

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
        labels=parse_category_labels(row.get("product_category"))
        + parse_category_labels(row.get("gift_themes")),
    )


def parse_category_labels(value: object) -> tuple[str, ...]:
    """카테고리·테마 컬럼 값을 라벨 목록으로 푼다.

    "감사|친구" 같은 구분자 문자열과 '["감사", "친구"]' 같은 JSON 배열(MySQL JSON 컬럼)을
    둘 다 받는다.
    """
    if value is None:
        return ()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, list | tuple):
        return tuple(str(label).strip() for label in value if str(label).strip())
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(label).strip() for label in parsed if str(label).strip())
    return tuple(label.strip() for label in text.split("|") if label.strip())
