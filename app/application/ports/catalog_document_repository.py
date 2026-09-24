from collections.abc import Sequence
from typing import Protocol

from app.application.catalog_document_service import ProductEnrichment
from app.domain.catalog.sync import SourceProduct
from app.domain.recommendation.models import ProductKey


class CatalogDocumentRepository(Protocol):
    """문서 생성 배치가 쓰는 저장소.

    claim -> save 또는 mark_failed 가 한 사이클이다. 배치가 중간에 죽어도
    release_stale 이 처리 중으로 남은 상품을 다시 대기 상태로 돌린다.
    """

    async def claim_pending(self, *, limit: int) -> Sequence[SourceProduct]:
        """pending 상품을 processing 으로 바꾸고 가져온다.

        여러 워커가 동시에 돌아도 같은 상품을 가져가지 않아야 한다.
        """
        ...

    async def save_documents(
        self,
        enrichment: ProductEnrichment,
        *,
        vectors: Sequence[Sequence[float]],
        embedding_model_version: str,
    ) -> None:
        """세 공간의 문서와 벡터를 저장하고 상품을 ready 로 바꾼다."""
        ...

    async def mark_failed(self, key: ProductKey, *, reason: str) -> None: ...

    async def requeue_failed(self) -> int:
        """활성 상품의 failed 상태를 pending 으로 되돌린다."""
        ...

    async def release_stale(self, *, older_than_seconds: int) -> int:
        """죽은 배치가 processing 으로 남긴 상품을 pending 으로 되돌린다."""
        ...
