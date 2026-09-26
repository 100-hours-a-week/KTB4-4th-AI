from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.application.catalog_document_service import CatalogDocumentService, ProductEnrichment
from app.application.ports.catalog_document_repository import CatalogDocumentRepository
from app.application.ports.embedder import Embedder
from app.domain.catalog.models import DocumentSourceProduct

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class EnrichmentReport:
    claimed: int
    succeeded: int
    failed: int
    released: int
    requeued: int = 0

    @property
    def exhausted(self) -> bool:
        """더 처리할 pending 이 없으면 True."""
        return self.claimed == 0


class CatalogEnrichmentService:
    """pending 상품을 가져와 LLM 문서 3개를 만들고 임베딩해 저장한다.

    동기화 배치와는 processing_status 컬럼으로만 연결된다. 서로를 직접
    호출하지 않으므로 어느 쪽이 먼저 돌든, 한쪽이 죽든 결과가 같다.
    """

    def __init__(
        self,
        *,
        documents: CatalogDocumentService,
        embedder: Embedder,
        repository: CatalogDocumentRepository,
        batch_size: int = 50,
        embedding_concurrency: int = 4,
        progress_interval: int = 100,
        stale_processing_seconds: int = 3600,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if embedding_concurrency <= 0:
            raise ValueError("embedding_concurrency must be positive")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be positive")
        self._documents = documents
        self._embedder = embedder
        self._repository = repository
        self._batch_size = batch_size
        self._embedding_semaphore = asyncio.Semaphore(embedding_concurrency)
        self._progress_interval = progress_interval
        self._stale_processing_seconds = stale_processing_seconds

    async def run_once(self) -> EnrichmentReport:
        """한 배치만 처리한다. 남은 것은 다음 실행이 이어받는다."""
        released = await self._repository.release_stale(
            older_than_seconds=self._stale_processing_seconds,
        )
        claimed = await self._repository.claim_pending(limit=self._batch_size)
        if not claimed:
            return EnrichmentReport(claimed=0, succeeded=0, failed=0, released=released)

        outcomes = await asyncio.gather(
            *(self._process(product.to_document_input()) for product in claimed)
        )
        succeeded = sum(outcomes)

        return EnrichmentReport(
            claimed=len(claimed),
            succeeded=succeeded,
            failed=len(claimed) - succeeded,
            released=released,
        )

    async def run_until_drained(
        self,
        *,
        max_batches: int = 100,
        retry_failed: bool = False,
    ) -> EnrichmentReport:
        """pending 이 바닥나거나 배치 상한에 닿을 때까지 반복한다."""
        requeued = await self._repository.requeue_failed() if retry_failed else 0
        totals = EnrichmentReport(
            claimed=0,
            succeeded=0,
            failed=0,
            released=0,
            requeued=requeued,
        )
        next_progress = self._progress_interval
        for _ in range(max_batches):
            report = await self.run_once()
            totals = EnrichmentReport(
                claimed=totals.claimed + report.claimed,
                succeeded=totals.succeeded + report.succeeded,
                failed=totals.failed + report.failed,
                released=totals.released + report.released,
                requeued=totals.requeued,
            )
            while totals.claimed >= next_progress:
                logger.info(
                    "catalog enrichment progress: claimed=%d succeeded=%d failed=%d",
                    totals.claimed,
                    totals.succeeded,
                    totals.failed,
                )
                next_progress += self._progress_interval
            if report.exhausted:
                break
        return totals

    async def _process(self, product: DocumentSourceProduct) -> bool:
        try:
            enrichment = await self._documents.generate(product)
        except Exception as error:  # 생성 실패는 상품 단위로 격리한다
            logger.warning(
                "document generation failed",
                extra={"product": str(product.key), "reason": str(error)},
            )
            await self._repository.mark_failed(product.key, reason=str(error))
            return False

        try:
            async with self._embedding_semaphore:
                await self._store(enrichment)
        except Exception as error:  # 임베딩이나 저장 실패는 상품 단위로 격리한다
            logger.warning(
                "embedding or persistence failed",
                extra={"product": str(enrichment.key), "reason": str(error)},
            )
            await self._repository.mark_failed(enrichment.key, reason=str(error))
            return False
        return True

    async def _store(self, enrichment: ProductEnrichment) -> None:
        # 문서 순서는 ProductDocumentSet 이 space 이름순으로 고정해 둔다.
        # 벡터와 문서가 어긋나면 추천이 조용히 틀리므로 순서를 그대로 쓴다.
        texts = enrichment.documents.ordered_texts
        vectors = await self._embedder.embed_passages(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedder returned an unexpected number of passage vectors")
        await self._repository.save_documents(
            enrichment,
            vectors=vectors,
            embedding_model_version=self._embedder.space_id,
        )
