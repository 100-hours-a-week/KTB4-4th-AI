from __future__ import annotations

import asyncio
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from app.application.catalog_document_service import (
    ProductEnrichment,
)
from app.application.catalog_enrichment_service import CatalogEnrichmentService
from app.domain.catalog import DocumentEvidence, ProductDocument, ProductDocumentSet
from app.domain.catalog.sync import SourceProduct
from app.domain.recommendation import ProductKey, VectorSpace


def _source(external_id: str) -> SourceProduct:
    return SourceProduct(
        key=ProductKey(platform="coupang", external_id=external_id),
        name=f"캠핑 용품 {external_id}",
        price=Decimal(30000),
    )


def _enrichment(key: ProductKey) -> ProductEnrichment:
    return ProductEnrichment(
        key=key,
        documents=ProductDocumentSet(
            key=key,
            documents=(
                ProductDocument(space=VectorSpace.CONTENT, text="티타늄 소재의 아웃도어 머그컵."),
                ProductDocument(space=VectorSpace.USAGE, text="캠핑에서 음료를 마실 때 쓰는 컵."),
                ProductDocument(space=VectorSpace.GIFT, text="캠핑을 즐기는 사람에게 좋은 선물."),
            ),
            evidence=DocumentEvidence.NAME_ONLY,
        ),
        category_code="hobby",
        category_confidence=0.9,
        price_band=None,
        attributes={},
    )


class FakeDocumentService:
    def __init__(self, *, fail_keys: set[str] | None = None) -> None:
        self.fail_keys = fail_keys or set()
        self.seen: list[str] = []

    async def generate(self, product: Any) -> ProductEnrichment:
        self.seen.append(product.key.external_id)
        if product.key.external_id in self.fail_keys:
            raise ValueError("bad json")
        return _enrichment(product.key)


class FakeEmbedder:
    def __init__(self, *, explode: bool = False, wrong_count: bool = False) -> None:
        self.explode = explode
        self.wrong_count = wrong_count
        self.batches: list[Sequence[str]] = []

    @property
    def space_id(self) -> str:
        return "solar-embedding-2"

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.explode:
            raise RuntimeError("embedding endpoint down")
        count = len(texts) - 1 if self.wrong_count else len(texts)
        return [[0.1, 0.2] for _ in range(count)]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class OverlapDocumentService:
    def __init__(self) -> None:
        self.embedding_started = asyncio.Event()
        self.second_generation_waiting = False

    async def generate(self, product: Any) -> ProductEnrichment:
        if product.key.external_id == "1":
            await asyncio.sleep(0)
            return _enrichment(product.key)
        self.second_generation_waiting = True
        await self.embedding_started.wait()
        self.second_generation_waiting = False
        return _enrichment(product.key)


class OverlapEmbedder(FakeEmbedder):
    def __init__(self, documents: OverlapDocumentService) -> None:
        super().__init__()
        self._documents = documents
        self.overlap_observed = False

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.overlap_observed |= self._documents.second_generation_waiting
        self._documents.embedding_started.set()
        return await super().embed_passages(texts)


class FakeRepository:
    def __init__(self, pages: list[list[SourceProduct]]) -> None:
        self.pages = pages
        self.saved: list[ProductEnrichment] = []
        self.failures: list[tuple[str, str]] = []
        self.released = 0
        self.requeued = 0

    async def claim_pending(self, *, limit: int) -> Sequence[SourceProduct]:
        return self.pages.pop(0) if self.pages else []

    async def save_documents(
        self,
        enrichment: ProductEnrichment,
        *,
        vectors: Sequence[Sequence[float]],
        embedding_model_version: str,
    ) -> None:
        assert len(vectors) == 3
        assert embedding_model_version == "solar-embedding-2"
        self.saved.append(enrichment)

    async def mark_failed(self, key: ProductKey, *, reason: str) -> None:
        self.failures.append((key.external_id, reason))

    async def requeue_failed(self) -> int:
        self.requeued += 1
        return 3

    async def release_stale(self, *, older_than_seconds: int) -> int:
        self.released += 1
        return 2


def _service(
    repository: FakeRepository,
    *,
    documents: FakeDocumentService | None = None,
    embedder: FakeEmbedder | None = None,
) -> CatalogEnrichmentService:
    return CatalogEnrichmentService(
        documents=documents or FakeDocumentService(),
        embedder=embedder or FakeEmbedder(),
        repository=repository,
        batch_size=10,
    )


def test_run_once_saves_three_documents_per_product() -> None:
    repository = FakeRepository([[_source("1"), _source("2")]])
    embedder = FakeEmbedder()
    service = _service(repository, embedder=embedder)

    report = asyncio.run(service.run_once())

    assert report.claimed == 2
    assert report.succeeded == 2
    assert report.failed == 0
    assert len(repository.saved) == 2
    # 상품당 문서 3개를 한 번에 임베딩한다.
    assert [len(batch) for batch in embedder.batches] == [3, 3]


def test_documents_are_embedded_in_stable_space_order() -> None:
    repository = FakeRepository([[_source("1")]])
    embedder = FakeEmbedder()
    service = _service(repository, embedder=embedder)

    asyncio.run(service.run_once())

    saved = repository.saved[0]
    assert [document.space for document in saved.documents.documents] == [
        VectorSpace.CONTENT,
        VectorSpace.GIFT,
        VectorSpace.USAGE,
    ]
    assert embedder.batches[0] == list(saved.documents.ordered_texts)


def test_generation_and_embedding_overlap() -> None:
    repository = FakeRepository([[_source("1"), _source("2")]])
    documents = OverlapDocumentService()
    embedder = OverlapEmbedder(documents)
    service = CatalogEnrichmentService(
        documents=documents,  # type: ignore[arg-type]
        embedder=embedder,
        repository=repository,
        batch_size=10,
    )

    report = asyncio.run(service.run_once())

    assert report.succeeded == 2
    assert embedder.overlap_observed


def test_llm_failure_is_isolated_to_one_product() -> None:
    repository = FakeRepository([[_source("1"), _source("2")]])
    service = _service(repository, documents=FakeDocumentService(fail_keys={"2"}))

    report = asyncio.run(service.run_once())

    assert report.succeeded == 1
    assert report.failed == 1
    assert repository.failures == [("2", "bad json")]
    assert len(repository.saved) == 1


def test_embedding_failure_marks_product_failed_not_ready() -> None:
    repository = FakeRepository([[_source("1")]])
    service = _service(repository, embedder=FakeEmbedder(explode=True))

    report = asyncio.run(service.run_once())

    assert report.succeeded == 0
    assert repository.saved == []
    assert repository.failures[0][0] == "1"


def test_vector_count_mismatch_is_rejected() -> None:
    repository = FakeRepository([[_source("1")]])
    service = _service(repository, embedder=FakeEmbedder(wrong_count=True))

    report = asyncio.run(service.run_once())

    assert report.succeeded == 0
    assert repository.saved == []


def test_empty_queue_reports_exhausted_and_still_releases_stale() -> None:
    repository = FakeRepository([])
    service = _service(repository)

    report = asyncio.run(service.run_once())

    assert report.exhausted
    assert report.claimed == 0
    assert report.released == 2


def test_run_until_drained_stops_when_queue_empties() -> None:
    repository = FakeRepository([[_source("1")], [_source("2")], []])
    service = _service(repository)

    report = asyncio.run(service.run_until_drained(max_batches=10))

    assert report.claimed == 2
    assert report.succeeded == 2


def test_run_until_drained_respects_batch_ceiling() -> None:
    repository = FakeRepository([[_source(str(index))] for index in range(10)])
    service = _service(repository)

    report = asyncio.run(service.run_until_drained(max_batches=3))

    assert report.claimed == 3


def test_run_until_drained_requeues_failed_only_when_requested() -> None:
    repository = FakeRepository([])
    service = _service(repository)

    report = asyncio.run(service.run_until_drained(retry_failed=True))

    assert repository.requeued == 1
    assert report.requeued == 3
