from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.application.catalog_document_prompts import JSON_SCHEMA, build_document_messages
from app.application.ports.model_gateway import ModelGateway
from app.domain.catalog.models import (
    DocumentEvidence,
    DocumentSourceProduct,
    DocumentValidationError,
    PriceBand,
    ProductDocument,
    ProductDocumentSet,
    price_band,
)
from app.domain.recommendation.models import ProductKey, VectorSpace

_SPACE_FIELDS = {
    VectorSpace.CONTENT: "contentDocument",
    VectorSpace.USAGE: "usageDocument",
    VectorSpace.GIFT: "giftDocument",
}


@dataclass(slots=True, frozen=True)
class ProductEnrichment:
    """LLM 한 번 호출로 얻은 검색 문서와 파생 피처."""

    key: ProductKey
    documents: ProductDocumentSet
    category_code: str
    category_confidence: float
    price_band: PriceBand | None
    attributes: Mapping[str, Any]


@dataclass(slots=True, frozen=True)
class DocumentGenerationFailure:
    key: ProductKey
    reason: str


@dataclass(slots=True, frozen=True)
class DocumentGenerationResult:
    enriched: tuple[ProductEnrichment, ...]
    failed: tuple[DocumentGenerationFailure, ...]


class CatalogDocumentService:
    """상품명과 가격만으로 content·usage·gift 검색 문서를 만든다.

    상세 페이지 본문 없이 만든 문서이므로 evidence 는 항상 name_only 이고,
    적재 파이프라인은 이 결과를 ready 가 아니라 draft 상태로 저장해야 한다.
    """

    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        model: str,
        concurrency: int = 4,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self._gateway = model_gateway
        self._model = model
        self._semaphore = asyncio.Semaphore(concurrency)

    async def generate(self, product: DocumentSourceProduct) -> ProductEnrichment:
        async with self._semaphore:
            payload = await self._gateway.structured(
                build_document_messages(product),
                model=self._model,
                json_schema=JSON_SCHEMA,
            )
        return self._to_enrichment(product, payload)

    async def generate_many(
        self,
        products: Sequence[DocumentSourceProduct],
    ) -> DocumentGenerationResult:
        outcomes = await asyncio.gather(
            *(self.generate(product) for product in products),
            return_exceptions=True,
        )
        enriched: list[ProductEnrichment] = []
        failed: list[DocumentGenerationFailure] = []
        for product, outcome in zip(products, outcomes, strict=True):
            if isinstance(outcome, ProductEnrichment):
                enriched.append(outcome)
            elif isinstance(outcome, Exception):
                failed.append(DocumentGenerationFailure(key=product.key, reason=str(outcome)))
            else:  # pragma: no cover - gather 는 결과 아니면 예외만 돌려준다
                raise TypeError("unexpected generation outcome")
        return DocumentGenerationResult(enriched=tuple(enriched), failed=tuple(failed))

    def _to_enrichment(
        self,
        product: DocumentSourceProduct,
        payload: Mapping[str, Any],
    ) -> ProductEnrichment:
        documents = []
        for space, field in _SPACE_FIELDS.items():
            text = payload.get(field)
            if not isinstance(text, str):
                raise DocumentValidationError(f"{field} is missing from the model response")
            documents.append(ProductDocument(space=space, text=text))

        category_code = payload.get("categoryCode")
        if not isinstance(category_code, str) or not category_code.strip():
            raise DocumentValidationError("categoryCode is missing from the model response")

        confidence = payload.get("categoryConfidence")
        if not isinstance(confidence, int | float) or isinstance(confidence, bool):
            raise DocumentValidationError("categoryConfidence must be a number")
        confidence = min(max(float(confidence), 0.0), 1.0)

        attributes = payload.get("attributes")
        if not isinstance(attributes, Mapping):
            raise DocumentValidationError("attributes must be an object")

        return ProductEnrichment(
            key=product.key,
            documents=ProductDocumentSet(
                key=product.key,
                documents=tuple(documents),
                evidence=DocumentEvidence.NAME_ONLY,
            ),
            category_code=category_code.strip(),
            category_confidence=confidence,
            price_band=price_band(product.price),
            attributes=dict(attributes),
        )
