from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.domain.catalog.labels import (
    CategorySource,
    GiftTheme,
    ProductCategory,
    SourceLabels,
    is_gift_available,
    split_labels,
)
from app.domain.catalog.models import DocumentSourceProduct, price_band
from app.domain.recommendation.models import ProductKey

_HASH_SEPARATOR = "\x1f"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class SyncMode(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"


@dataclass(slots=True, frozen=True)
class SourceProduct:
    """원본 상품 한 행. 여기 있는 값만 AI DB가 원본으로 인정한다.

    labels 는 원본 product_category 그대로다. 카테고리 하나와 선물 테마가 섞여 있다.
    """

    key: ProductKey
    name: str
    price: Decimal | None
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = " ".join(self.name.split())
        if not name:
            raise ValueError("product name must not be empty")
        if self.price is not None and self.price < 0:
            raise ValueError("product price must not be negative")
        object.__setattr__(self, "name", name)
        labels = tuple(
            dict.fromkeys(" ".join(label.split()) for label in self.labels if label.strip())
        )
        object.__setattr__(self, "labels", labels)

    @property
    def source_labels(self) -> SourceLabels:
        return split_labels(self.labels)

    @property
    def themes(self) -> tuple[GiftTheme, ...]:
        return self.source_labels.themes

    @property
    def category_assignment(self) -> tuple[ProductCategory | None, CategorySource | None]:
        """원본 카테고리. 없으면 문서 생성 모델이 고르도록 비워 둔다."""
        category = self.source_labels.category
        if category is not None:
            return category, CategorySource.SOURCE
        return None, None

    @property
    def gift_available(self) -> bool:
        return is_gift_available(self.category_assignment[0], self.themes)

    @property
    def source_hash(self) -> str:
        """원본 값 전체의 해시. 달라지면 스냅샷을 갱신한다."""
        price = "" if self.price is None else format(self.price.normalize(), "f")
        return _digest(
            self.key.platform,
            self.key.external_id,
            self.name,
            price,
            "|".join(sorted(self.labels)),
        )

    @property
    def doc_input_hash(self) -> str:
        """문서 생성에 실제로 쓰는 값만의 해시. 달라질 때만 LLM을 다시 부른다.

        가격은 금액이 아니라 구간으로 들어간다. 그래서 같은 구간 안에서
        가격만 오르내리면 문서와 벡터를 다시 만들지 않는다.
        """
        band = price_band(self.price)
        category, _ = self.category_assignment
        return _digest(
            self.name,
            band.value if band is not None else "",
            category.value if category is not None else "",
            ",".join(theme.value for theme in self.themes),
        )

    def to_document_input(self, *, subcategory: str | None = None) -> DocumentSourceProduct:
        """subcategory 는 원본 값이 아니라 AI DB에 저장된 판단 결과라 따로 받는다."""
        category, source = self.category_assignment
        return DocumentSourceProduct(
            key=self.key,
            name=self.name,
            price=self.price,
            category=category,
            category_source=source,
            themes=self.themes,
            subcategory=subcategory,
        )


def _digest(*parts: str) -> str:
    return hashlib.sha256(_HASH_SEPARATOR.join(parts).encode("utf-8")).hexdigest()
