from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal

import pytest

from app.application.catalog_sync_service import CatalogSyncService
from app.domain.catalog.sync import ProcessingStatus, SourceProduct, SyncMode
from app.domain.recommendation import ProductKey


def _product(external_id: str, name: str, price: str | None = "10000") -> SourceProduct:
    return SourceProduct(
        key=ProductKey(platform="coupang", external_id=external_id),
        name=name,
        price=None if price is None else Decimal(price),
    )


class FakeSource:
    def __init__(self, batches: list[list[SourceProduct]]) -> None:
        self._batches = batches
        self.requested_batch_size: int | None = None

    async def ping(self) -> bool:
        return True

    async def iter_products(
        self,
        *,
        batch_size: int,
    ) -> AsyncIterator[Sequence[SourceProduct]]:
        self.requested_batch_size = batch_size
        for batch in self._batches:
            yield batch


class FakeRepository:
    def __init__(self, *, changed_per_batch: int = 1) -> None:
        self.changed_per_batch = changed_per_batch
        self.upserted: list[Sequence[SourceProduct]] = []
        self.deactivate_calls = 0
        self.finished: dict[str, object] | None = None

    async def start_run(self, *, mode: SyncMode) -> int:
        self.mode = mode
        return 7

    async def upsert_products(
        self,
        products: Sequence[SourceProduct],
        *,
        run_id: int,
    ) -> int:
        self.upserted.append(products)
        return self.changed_per_batch

    async def deactivate_missing(self, *, run_id: int) -> int:
        self.deactivate_calls += 1
        return 3

    async def finish_run(
        self,
        *,
        run_id: int,
        received_count: int,
        changed_count: int,
        deactivated_count: int,
        failed: bool = False,
    ) -> None:
        self.finished = {
            "run_id": run_id,
            "received": received_count,
            "changed": changed_count,
            "deactivated": deactivated_count,
            "failed": failed,
        }


def test_full_sync_reports_counts_and_deactivates() -> None:
    source = FakeSource(
        [[_product("1", "캠핑 랜턴"), _product("2", "머그컵")], [_product("3", "텐트")]]
    )
    repository = FakeRepository()
    service = CatalogSyncService(source=source, repository=repository, batch_size=500)

    report = asyncio.run(service.sync())

    assert report.run_id == 7
    assert report.received_count == 3
    assert report.changed_count == 2
    assert report.deactivated_count == 3
    assert repository.deactivate_calls == 1
    assert source.requested_batch_size == 500
    assert repository.finished == {
        "run_id": 7,
        "received": 3,
        "changed": 2,
        "deactivated": 3,
        "failed": False,
    }


def test_incremental_sync_does_not_deactivate() -> None:
    source = FakeSource([[_product("1", "캠핑 랜턴")]])
    repository = FakeRepository()
    service = CatalogSyncService(source=source, repository=repository)

    report = asyncio.run(service.sync(mode=SyncMode.INCREMENTAL))

    assert repository.deactivate_calls == 0
    assert report.deactivated_count == 0


def test_failure_marks_run_failed_and_reraises() -> None:
    class ExplodingRepository(FakeRepository):
        async def upsert_products(
            self,
            products: Sequence[SourceProduct],
            *,
            run_id: int,
        ) -> int:
            raise RuntimeError("postgres down")

    source = FakeSource([[_product("1", "캠핑 랜턴")]])
    repository = ExplodingRepository()
    service = CatalogSyncService(source=source, repository=repository)

    with pytest.raises(RuntimeError):
        asyncio.run(service.sync())

    assert repository.finished is not None
    assert repository.finished["failed"] is True


def test_source_hash_changes_with_price_but_doc_hash_does_not() -> None:
    before = _product("1", "티타늄 머그컵", "48000")
    after = _product("1", "티타늄 머그컵", "45000")

    assert before.source_hash != after.source_hash
    # 둘 다 mid 구간이라 문서를 다시 만들 이유가 없다.
    assert before.doc_input_hash == after.doc_input_hash


def test_doc_hash_changes_when_price_crosses_band() -> None:
    mid = _product("1", "티타늄 머그컵", "48000")
    high = _product("1", "티타늄 머그컵", "180000")

    assert mid.doc_input_hash != high.doc_input_hash


def test_doc_hash_changes_when_name_changes() -> None:
    before = _product("1", "티타늄 머그컵", "48000")
    after = _product("1", "스테인리스 머그컵", "48000")

    assert before.doc_input_hash != after.doc_input_hash


def test_processing_status_values_match_migration() -> None:
    assert [status.value for status in ProcessingStatus] == [
        "pending",
        "processing",
        "ready",
        "failed",
    ]
