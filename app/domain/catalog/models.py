from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.domain.catalog.labels import (
    CategorySource,
    GiftTheme,
    ProductCategory,
    is_gift_available,
)
from app.domain.catalog.subcategories import subcategory_label
from app.domain.recommendation.models import ProductKey, VectorSpace

# 문서 생성 규칙 버전. 프롬프트나 아래 검증 규칙을 바꾸면 반드시 올린다.
DOCUMENT_VERSION = "catalog-doc-v4-subcategory"

# 모든 상품이 갖는 공간. gift 는 gift_available 상품에만 만든다.
REQUIRED_SPACES = frozenset({VectorSpace.CONTENT, VectorSpace.USAGE})

MAX_DOCUMENT_LENGTH = 200
MIN_DOCUMENT_LENGTH = 10

# 문서에 값이 자주 바뀌는 정보가 섞이면 가격이 바뀔 때마다 재임베딩해야 한다.
_PRICE_PATTERN = re.compile(r"\d[\d,]*\s*(원|만원|₩|won|KRW)", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


class DocumentEvidence(StrEnum):
    """문서가 어떤 근거로 쓰였는지. 상세 본문 없이 쓴 문서는 ready로 승격하지 않는다."""

    NAME_ONLY = "name_only"
    SOURCE_CONTENT = "source_content"


class PriceBand(StrEnum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class DocumentValidationError(ValueError):
    """LLM이 만든 검색 문서가 임베딩에 쓸 수 없는 형태일 때."""


@dataclass(slots=True, frozen=True)
class DocumentSourceProduct:
    """문서 생성 입력. 상품명·가격과 원본 product_category 의 카테고리·선물 테마를 쓴다.

    category 가 None 이면 원본에 카테고리가 없는 상품이라 문서 생성 모델이 11개 중에서 고른다.
    subcategory 는 판단 모델이 고른 세부 카테고리다. 확신이 낮거나 아직 분류 전이면 None 이다.
    """

    key: ProductKey
    name: str
    price: Decimal | None = None
    category: ProductCategory | None = None
    category_source: CategorySource | None = None
    themes: tuple[GiftTheme, ...] = ()
    subcategory: str | None = None

    def __post_init__(self) -> None:
        name = " ".join(self.name.split())
        if not name:
            raise ValueError("product name must not be empty")
        if self.price is not None and self.price < 0:
            raise ValueError("product price must not be negative")
        if (self.category is None) != (self.category_source is None):
            raise ValueError("category and category_source must be set together")
        if self.category_source is CategorySource.INFERRED:
            raise ValueError("an inferred category comes from document generation, not the input")
        if self.subcategory is not None:
            if self.category is None:
                raise ValueError("subcategory needs a category")
            subcategory_label(self.category, self.subcategory)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "themes", tuple(theme for theme in GiftTheme if theme in self.themes)
        )

    @property
    def gift_available(self) -> bool:
        return is_gift_available(self.category, self.themes)

    @property
    def subcategory_label(self) -> str | None:
        if self.category is None or self.subcategory is None:
            return None
        return subcategory_label(self.category, self.subcategory)


@dataclass(slots=True, frozen=True)
class ProductDocument:
    space: VectorSpace
    text: str

    def __post_init__(self) -> None:
        text = " ".join(self.text.split())
        validate_document_text(self.space, text)
        object.__setattr__(self, "text", text)

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(slots=True, frozen=True)
class ProductDocumentSet:
    key: ProductKey
    documents: tuple[ProductDocument, ...]
    evidence: DocumentEvidence
    document_version: str = DOCUMENT_VERSION

    def __post_init__(self) -> None:
        spaces = [document.space for document in self.documents]
        if len(set(spaces)) != len(spaces):
            raise DocumentValidationError("each space may appear only once in a document set")
        if not REQUIRED_SPACES <= set(spaces):
            raise DocumentValidationError("document set must include content and usage")
        texts = [document.text for document in self.documents]
        if len(set(texts)) != len(texts):
            raise DocumentValidationError("content, usage, and gift documents must differ")
        object.__setattr__(
            self,
            "documents",
            tuple(sorted(self.documents, key=lambda document: document.space.value)),
        )

    @property
    def spaces(self) -> tuple[VectorSpace, ...]:
        return tuple(document.space for document in self.documents)

    def text_for(self, space: VectorSpace) -> str:
        return next(document.text for document in self.documents if document.space == space)

    @property
    def ordered_texts(self) -> tuple[str, ...]:
        return tuple(document.text for document in self.documents)


def validate_document_text(space: VectorSpace, text: str) -> None:
    if not text.strip():
        raise DocumentValidationError(f"{space.value} document must not be empty")
    if len(text) < MIN_DOCUMENT_LENGTH:
        raise DocumentValidationError(f"{space.value} document is too short to embed")
    if len(text) > MAX_DOCUMENT_LENGTH:
        raise DocumentValidationError(
            f"{space.value} document exceeds {MAX_DOCUMENT_LENGTH} characters"
        )
    if _PRICE_PATTERN.search(text):
        raise DocumentValidationError(f"{space.value} document must not contain a price")
    if _URL_PATTERN.search(text):
        raise DocumentValidationError(f"{space.value} document must not contain a URL")


def price_band(price: Decimal | None) -> PriceBand | None:
    """카테고리 통계가 생기기 전까지 쓰는 절대 금액 기준 임시 구간."""
    if price is None:
        return None
    if price < Decimal(20000):
        return PriceBand.LOW
    if price < Decimal(80000):
        return PriceBand.MID
    return PriceBand.HIGH
