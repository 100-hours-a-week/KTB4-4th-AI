from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

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
    """MySQL 원본 한 행. 여기 있는 값만 AI DB가 원본으로 인정한다."""

    key: ProductKey
    name: str
    price: Decimal | None

    def __post_init__(self) -> None:
        name = " ".join(self.name.split())
        if not name:
            raise ValueError("product name must not be empty")
        if self.price is not None and self.price < 0:
            raise ValueError("product price must not be negative")
        object.__setattr__(self, "name", name)

    @property
    def source_hash(self) -> str:
        """원본 4개 값 전체의 해시. 달라지면 스냅샷을 갱신한다."""
        price = "" if self.price is None else format(self.price.normalize(), "f")
        return _digest(self.key.platform, self.key.external_id, self.name, price)

    @property
    def doc_input_hash(self) -> str:
        """문서 생성에 실제로 쓰는 값만의 해시. 달라질 때만 LLM을 다시 부른다.

        가격은 금액이 아니라 구간으로 들어간다. 그래서 같은 구간 안에서
        가격만 오르내리면 문서와 벡터를 다시 만들지 않는다.
        """
        band = price_band(self.price)
        return _digest(self.name, band.value if band is not None else "")

    def to_document_input(self) -> DocumentSourceProduct:
        return DocumentSourceProduct(key=self.key, name=self.name, price=self.price)


def _digest(*parts: str) -> str:
    return hashlib.sha256(_HASH_SEPARATOR.join(parts).encode("utf-8")).hexdigest()
