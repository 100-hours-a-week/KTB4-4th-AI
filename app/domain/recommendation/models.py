from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from app.domain.profile.models import DeferralReason, TasteField, Visibility


class RecommendationMode(StrEnum):
    SELF = "self"
    GIFT = "gift"


class VectorSpace(StrEnum):
    CONTENT = "content"
    USAGE = "usage"
    GIFT = "gift"


@dataclass(slots=True, frozen=True)
class ProductKey:
    platform: str
    external_id: str

    def __post_init__(self) -> None:
        platform = self.platform.strip().lower()
        external_id = self.external_id.strip()
        if not platform or not external_id:
            raise ValueError("platform and external_id must not be empty")
        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "external_id", external_id)


@dataclass(slots=True, frozen=True)
class CatalogProduct:
    key: ProductKey
    name: str
    price: Decimal | None

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("product name must not be empty")
        if self.price is not None and self.price < 0:
            raise ValueError("product price must not be negative")
        object.__setattr__(self, "name", name)


@dataclass(slots=True, frozen=True)
class RecommendationSignal:
    field: TasteField
    value: str
    confidence: float
    visibility: Visibility
    updated_at: datetime
    deferral_reason: DeferralReason | None = None

    def __post_init__(self) -> None:
        value = self.value.strip()
        if not value:
            raise ValueError("recommendation signal value must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("recommendation signal confidence must be between 0 and 1")
        object.__setattr__(self, "value", value)


@dataclass(slots=True, frozen=True)
class SearchQuery:
    query_id: str
    signal: RecommendationSignal
    modes: frozenset[RecommendationMode]
    space: VectorSpace
    text: str


@dataclass(slots=True, frozen=True)
class VectorMatch:
    query: SearchQuery
    product: CatalogProduct
    similarity: float


@dataclass(slots=True, frozen=True)
class CatalogSearchResult:
    product: CatalogProduct
    similarity: float


@dataclass(slots=True, frozen=True)
class ScoredMatch:
    match: VectorMatch
    contribution: float


@dataclass(slots=True, frozen=True)
class RankedRecommendation:
    product: CatalogProduct
    mode: RecommendationMode
    rank: int
    score: float
    reason: str
    matches: tuple[ScoredMatch, ...]
