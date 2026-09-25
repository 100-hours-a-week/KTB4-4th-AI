from __future__ import annotations

import logging
from dataclasses import dataclass

from app.application.ports.catalog_sync_repository import CatalogSyncRepository
from app.application.ports.product_source import ProductSource
from app.domain.catalog.sync import SyncMode

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SyncReport:
    run_id: int
    received_count: int
    changed_count: int
    deactivated_count: int


class CatalogSyncService:
    """MySQL 원본을 AI 카탈로그 스냅샷에 반영한다.

    이 단계는 가공을 하지 않는다. 문서 생성과 임베딩은 여기서 pending 으로
    표시된 상품을 나중에 별도 배치가 가져가서 처리한다. 그래야 느리고 자주
    실패하는 LLM 작업이 원본 DB 커넥션을 붙잡지 않는다.
    """

    def __init__(
        self,
        *,
        source: ProductSource,
        repository: CatalogSyncRepository,
        batch_size: int = 1000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._source = source
        self._repository = repository
        self._batch_size = batch_size

    async def sync(self, *, mode: SyncMode = SyncMode.FULL) -> SyncReport:
        run_id = await self._repository.start_run(mode=mode)
        received = 0
        changed = 0
        deactivated = 0
        try:
            async for batch in self._source.iter_products(batch_size=self._batch_size):
                if not batch:
                    continue
                received += len(batch)
                changed += await self._repository.upsert_products(batch, run_id=run_id)
                logger.info(
                    "catalog sync progress",
                    extra={"run_id": run_id, "received": received, "changed": changed},
                )
            # 증분 실행은 전체를 훑지 않으므로 사라진 상품을 판단할 수 없다.
            if mode is SyncMode.FULL:
                deactivated = await self._repository.deactivate_missing(run_id=run_id)
        except Exception:
            await self._repository.finish_run(
                run_id=run_id,
                received_count=received,
                changed_count=changed,
                deactivated_count=deactivated,
                failed=True,
            )
            raise

        await self._repository.finish_run(
            run_id=run_id,
            received_count=received,
            changed_count=changed,
            deactivated_count=deactivated,
        )
        return SyncReport(
            run_id=run_id,
            received_count=received,
            changed_count=changed,
            deactivated_count=deactivated,
        )
