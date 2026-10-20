from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from app.domain.profile.models import DeferralReason, PreferenceAspect, TasteField, Visibility
from app.domain.recommendation.exclusions import CatalogExclusion


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
    # 카탈로그 카테고리 코드(app.domain.catalog.labels.ProductCategory 값).
    # catalog 도메인이 이 모듈을 import 하므로 순환을 피하려고 문자열로 둔다.
    category: str | None = None
    # 선물 목록(gift) 후보인지. 테마가 붙었거나 교환권·명품·패션인 상품이다.
    gift_available: bool = False
    # 세부 카테고리 코드(app.domain.catalog.subcategories). 판단 모델의 확신이 낮으면 None 이다.
    subcategory: str | None = None
    # content 문서 원문. 온보딩 제외 조건의 단어 필터에 쓴다.
    content_document: str | None = None

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
    # 대화에서 매긴 취향 순위 점수. 없으면 신뢰도를 그대로 가중치로 쓴다.
    rank_score: float | None = None
    # preferences 신호에만 쓴다. situation 취향은 보조 usage 쿼리가 된다.
    aspect: PreferenceAspect | None = None
    # 취향이 붙은 관심사(커피, 캠핑). 없으면 여러 관심사를 가로지르는 취향이다.
    target: str | None = None
    # 취향 축이나 관심사 분야. 같은 분류에 쿼리가 몰리지 않게 하는 데 쓴다.
    taxonomy_path: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        value = self.value.strip()
        if not value:
            raise ValueError("recommendation signal value must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("recommendation signal confidence must be between 0 and 1")
        if self.rank_score is not None and not 0.0 <= self.rank_score <= 1.0:
            raise ValueError("recommendation signal rank_score must be between 0 and 1")
        object.__setattr__(self, "value", value)

    @property
    def weight(self) -> float:
        return self.confidence if self.rank_score is None else self.rank_score


@dataclass(slots=True, frozen=True)
class RecommendationGuide:
    """쿼리를 만들지는 않고 후보를 거르거나 가산하는 데만 쓰는 취향 정보."""

    exclusions: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    # 온보딩의 알레르기·선물 제외 조건. exclusions 와 달리 유사도가 아니라 정확히 거른다.
    catalog_exclusion: CatalogExclusion = CatalogExclusion()


@dataclass(slots=True, frozen=True)
class SearchQuery:
    query_id: str
    signal: RecommendationSignal
    modes: frozenset[RecommendationMode]
    space: VectorSpace
    text: str
    # 카탈로그에 그 관심사의 상품이 없을 때 대신 찾는 탐색 쿼리. 추천 이유와 자리 배분이 다르다.
    exploration: bool = False
    # 취향·상황 쿼리를 짝지은 관심사("캠핑 관련 제품 중 미니멀한 디자인"의 캠핑).
    # 그 관심사가 카탈로그에 없다고 판정되면 이 쿼리도 함께 버린다.
    anchor: RecommendationSignal | None = None


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
