from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from app.domain.catalog.sync import SourceProduct


class ProductSource(Protocol):
    """상품 원본(백엔드 MySQL)에서 읽기만 한다."""

    async def ping(self) -> bool: ...

    def iter_products(self, *, batch_size: int) -> AsyncIterator[Sequence[SourceProduct]]:
        """키셋 페이징으로 전체 상품을 배치 단위로 흘려보낸다."""
        ...
