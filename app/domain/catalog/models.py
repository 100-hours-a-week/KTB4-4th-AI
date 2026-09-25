from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.domain.recommendation.models import ProductKey, VectorSpace

# 문서 생성 규칙 버전. 프롬프트나 아래 검증 규칙을 바꾸면 반드시 올린다.
DOCUMENT_VERSION = "catalog-doc-v1-name-price"

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
    """문서 생성 입력. 지금은 MySQL 원본에서 오는 상품명과 가격만 쓴다."""

    key: ProductKey
    name: str
    price: Decimal | None = None

    def __post_init__(self) -> None:
        name = " ".join(self.name.split())
        if not name:
            raise ValueError("product name must not be empty")
        if self.price is not None and self.price < 0:
            raise ValueError("product price must not be negative")
        object.__setattr__(self, "name", name)


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
        if set(spaces) != set(VectorSpace):
            raise DocumentValidationError(
                "document set must cover content, usage, and gift exactly once"
            )
        texts = [document.text for document in self.documents]
        if len(set(texts)) != len(texts):
            raise DocumentValidationError("content, usage, and gift documents must differ")
        object.__setattr__(
            self,
            "documents",
            tuple(sorted(self.documents, key=lambda document: document.space.value)),
        )

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
