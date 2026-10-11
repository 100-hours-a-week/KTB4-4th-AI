from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.application.catalog_document_prompts import build_document_messages, build_json_schema
from app.application.ports.model_gateway import ModelGateway
from app.domain.catalog.labels import (
    CATEGORY_LABELS,
    CategorySource,
    GiftTheme,
    ProductCategory,
)
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

_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
_CATEGORY_BY_LABEL = {label: category for category, label in CATEGORY_LABELS.items()}
# gift 문서는 받는 사람을 취향으로만 쓴다. 관계·성별이 벡터에 들어가면
# "여자친구 선물" 같은 상품끼리 뭉쳐서 취향 차이가 묻힌다.
_GIFT_RELATION_PATTERN = re.compile(
    r"남자\s?친구|여자\s?친구|남친|여친|연인|커플|부모님|효도|아내|남편|와이프|"
    r"(?:남자|여자|남성|여성)(?:\s?(?:에게|한테|분께|분에게|선물))"
)
_MATERIAL_TERMS = (
    "티타늄",
    "스테인리스",
    "알루미늄",
    "세라믹",
    "도자기",
    "실리콘",
    "플라스틱",
    "아크릴",
    "유리",
    "가죽",
    "원목",
    "나무",
    "고무",
    "황동",
    "구리",
    "순금",
    "14k",
    "18k",
    "24k",
    "순은",
    "실버",
    "코튼",
    "메리노울",
    "캐시미어",
    "폴리에스터",
    "나일론",
)


@dataclass(slots=True, frozen=True)
class ProductEnrichment:
    """LLM 한 번 호출로 얻은 검색 문서와 파생 피처."""

    key: ProductKey
    documents: ProductDocumentSet
    category: ProductCategory
    category_source: CategorySource
    # 모델이 고른 카테고리만 확신도가 있다. 원본 카테고리는 None.
    category_confidence: float | None
    themes: tuple[GiftTheme, ...]
    gift_available: bool
    price_band: PriceBand | None
    attributes: Mapping[str, Any]


class CatalogDocumentService:
    """상품명·가격·카테고리로 content·usage 검색 문서를, gift_available 상품에는 gift 문서도 만든다.

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
                json_schema=build_json_schema(product),
            )
        return self._to_enrichment(product, payload)

    def _to_enrichment(
        self,
        product: DocumentSourceProduct,
        payload: Mapping[str, Any],
    ) -> ProductEnrichment:
        gift_available = product.gift_available
        spaces = [VectorSpace.CONTENT, VectorSpace.USAGE]
        if gift_available:
            spaces.append(VectorSpace.GIFT)
        documents = []
        for space in spaces:
            field = _SPACE_FIELDS[space]
            text = payload.get(field)
            if not isinstance(text, str):
                raise DocumentValidationError(f"{field} is missing from the model response")
            # 판매 문구의 숫자("3%적립")까지 허용하도록 원본 이름과 비교한다.
            unsupported_numbers = _numbers(text) - _numbers(product.name)
            if unsupported_numbers:
                raise DocumentValidationError(
                    f"{field} contains numeric facts absent from the product name"
                )
            if space is VectorSpace.GIFT and _GIFT_RELATION_PATTERN.search(text):
                raise DocumentValidationError(
                    "giftDocument must describe the recipient by taste, not relation or gender"
                )
            documents.append(ProductDocument(space=space, text=text))

        category, source, confidence = self._category(product, payload)

        attributes = payload.get("attributes")
        if not isinstance(attributes, Mapping):
            raise DocumentValidationError("attributes must be an object")
        normalized_attributes = dict(attributes)
        normalized_attributes["materials"] = _materials_from_name(product.name)

        return ProductEnrichment(
            key=product.key,
            documents=ProductDocumentSet(
                key=product.key,
                documents=tuple(documents),
                evidence=DocumentEvidence.NAME_ONLY,
            ),
            category=category,
            category_source=source,
            category_confidence=confidence,
            themes=product.themes,
            gift_available=gift_available,
            price_band=price_band(product.price),
            attributes=normalized_attributes,
        )

    @staticmethod
    def _category(
        product: DocumentSourceProduct,
        payload: Mapping[str, Any],
    ) -> tuple[ProductCategory, CategorySource, float | None]:
        # 원본 카테고리는 모델 응답으로 바꾸지 않는다.
        if product.category is not None and product.category_source is not None:
            return product.category, product.category_source, None
        label = payload.get("category")
        category = _CATEGORY_BY_LABEL.get(label) if isinstance(label, str) else None
        if category is None:
            raise DocumentValidationError("category must be one of the 11 catalog categories")
        confidence = payload.get("categoryConfidence")
        if not isinstance(confidence, int | float) or isinstance(confidence, bool):
            raise DocumentValidationError("categoryConfidence must be a number")
        return category, CategorySource.INFERRED, min(max(float(confidence), 0.0), 1.0)


def _numbers(text: str) -> set[str]:
    return {match.group().replace(",", "") for match in _NUMBER_PATTERN.finditer(text)}


def _materials_from_name(name: str) -> list[str]:
    normalized_name = name.casefold()
    return [material for material in _MATERIAL_TERMS if material.casefold() in normalized_name]
