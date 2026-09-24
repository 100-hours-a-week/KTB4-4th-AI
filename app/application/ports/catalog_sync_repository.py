from collections.abc import Sequence
from typing import Protocol

from app.domain.catalog.sync import SourceProduct, SyncMode


class CatalogSyncRepository(Protocol):
    """AI 카탈로그(PostgreSQL) 쓰기. 원본 스냅샷 적재만 담당한다."""

    async def start_run(self, *, mode: SyncMode) -> int: ...

    async def upsert_products(
        self,
        products: Sequence[SourceProduct],
        *,
        run_id: int,
    ) -> int:
        """배치를 upsert하고 실제로 바뀐 행 수를 돌려준다."""
        ...

    async def deactivate_missing(self, *, run_id: int) -> int:
        """이번 실행에서 보이지 않은 상품을 비활성으로 내린다. 행은 지우지 않는다."""
        ...

    async def finish_run(
        self,
        *,
        run_id: int,
        received_count: int,
        changed_count: int,
        deactivated_count: int,
        failed: bool = False,
    ) -> None: ...
