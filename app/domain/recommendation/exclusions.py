"""온보딩의 알레르기와 선물 제외 조건을 카탈로그 필터로 바꾼다.

사용자가 직접 고른 조건이라 유사도로 걸러서는 안 된다. 두 단계로 거른다.

1. 구조 필터: 대분류와 세부 카테고리로 검색 SQL에서 뺀다. 후보 수(top_k)를 걸러질 상품에
   쓰지 않는다.
2. 단어 필터: 세부 카테고리로 잡히지 않는 상품(케이크가 든 카페 교환권, 와인잔)을 상품명과
   content 문서의 단어로 뺀다.

카탈로그에는 원재료·알레르기 표시가 없다. 그래서 알레르기는 그 성분이 흔히 들어가는 세부
카테고리를 통째로 빼고, 세부 카테고리가 없는 먹거리도 뺀다. 표시가 없다고 안전하다고 보지 않는다.
단어는 짧은 말이 다른 단어에 섞여 오탐이 나는 쪽(새우살, 대구축산)을 감수한다.

카테고리 코드는 app.domain.catalog.labels / subcategories 의 값이다. catalog 도메인이
recommendation 을 import 하므로 순환을 피하려고 문자열로 쓴다.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.domain.profile.onboarding import Allergy, GiftExclusion

if TYPE_CHECKING:
    # models 가 RecommendationGuide 에 이 모듈의 CatalogExclusion 을 담는다.
    from app.domain.recommendation.models import CatalogProduct

# 반려동물 대분류. 반려동물과 관련된 신호가 없는 사용자에게는 이 대분류를 통째로 뺀다.
# "산책"처럼 사람도 하는 활동이 강아지 산책용품에 걸리는 것을 막는다.
PET_CATEGORY = "pet"
# 사람이 먹는 상품. 반려동물 간식은 넣지 않는다.
_EDIBLE_CATEGORIES = frozenset({"food", "health"})
_EDIBLE_VOUCHER_SUBCATEGORIES = frozenset({"cafe_drink", "dessert", "meal", "convenience"})
# 세부 카테고리가 없는 먹거리. 무엇인지 모르므로 알레르기가 하나라도 있으면 뺀다.
_UNCLASSIFIED_EDIBLE = frozenset({"food:", "health:", "voucher:"})

_BAKED = frozenset({"food:bakery", "food:sweets", "voucher:dessert"})
_ALLERGY_SUBCATEGORIES: dict[Allergy, frozenset[str]] = {
    Allergy.PEANUT: _BAKED | {"health:sports_nutrition", "voucher:convenience"},
    Allergy.NUTS: _BAKED | {"health:sports_nutrition", "voucher:convenience"},
    Allergy.MILK: _BAKED | {"health:sports_nutrition", "voucher:convenience"},
    Allergy.EGG: _BAKED | {"voucher:convenience"},
    Allergy.WHEAT: _BAKED | {"food:pantry", "voucher:meal", "voucher:convenience"},
    Allergy.SOY: frozenset({"food:pantry", "health:sports_nutrition"}),
    Allergy.SHELLFISH: frozenset({"food:meat_seafood", "food:pantry"}),
    Allergy.FISH: frozenset({"food:meat_seafood", "food:pantry"}),
}
# 먹거리 상품의 상품명과 content 문서에서 찾는다. "넛"은 도넛에, "면"은 면세에 걸려서 쓰지 않는다.
_ALLERGY_TERMS: dict[Allergy, tuple[str, ...]] = {
    Allergy.PEANUT: ("땅콩", "피넛", "peanut"),
    Allergy.NUTS: (
        "견과",
        "넛츠",
        "아몬드",
        "호두",
        "캐슈",
        "피스타치오",
        "마카다미아",
        "헤이즐넛",
        "피칸",
        "브라질넛",
        "잣",
        "누텔라",
        "프랄린",
    ),
    Allergy.MILK: (
        "우유",
        "밀크",
        "치즈",
        "버터",
        "크림",
        "요거트",
        "요구르트",
        "유제품",
        "유청",
        "웨이프로틴",
        "라떼",
        "연유",
        "아이스크림",
        "프라페",
        "카스텔라",
    ),
    Allergy.EGG: ("계란", "달걀", "에그", "마요", "카스텔라", "머랭", "마카롱"),
    Allergy.WHEAT: (
        "빵",
        "케이크",
        "쿠키",
        "과자",
        "파스타",
        "베이글",
        "도넛",
        "와플",
        "크로와상",
        "마들렌",
        "휘낭시에",
        "타르트",
        "스콘",
        "카스텔라",
    ),
    Allergy.SOY: ("대두", "콩", "두유", "두부", "된장", "간장", "청국장", "낫또", "소이"),
    Allergy.SHELLFISH: (
        "갑각",
        "새우",
        "쉬림프",
        "슈림프",
        "꽃게",
        "대게",
        "홍게",
        "킹크랩",
        "크랩",
        "랍스터",
        "가재",
        "게장",
    ),
    Allergy.FISH: (
        "생선",
        "피쉬",
        "피시",
        "연어",
        "참치",
        "고등어",
        "굴비",
        "조기",
        "갈치",
        "장어",
        "명태",
        "황태",
        "멸치",
        "어묵",
        "민어",
        "광어",
        "우럭",
        "도미",
    ),
}

_GIFT_EXCLUSION_CATEGORIES: dict[GiftExclusion, frozenset[str]] = {
    # 건강식품(홍삼, 영양제)도 받는 쪽에서는 먹는 선물이다.
    GiftExclusion.FOOD: frozenset({"food", "health"}),
    GiftExclusion.VOUCHER: frozenset({"voucher"}),
}
_GIFT_EXCLUSION_SUBCATEGORIES: dict[GiftExclusion, frozenset[str]] = {
    GiftExclusion.PERFUME: frozenset({"beauty:fragrance"}),
    GiftExclusion.CLOTHING: frozenset(
        {
            "fashion:apparel",
            "fashion:loungewear",
            "luxury:apparel",
            "luxury:loungewear",
            "sports:sportswear",
        }
    ),
    # 향수는 따로 고르는 항목이라 화장품에 넣지 않는다.
    GiftExclusion.COSMETICS: frozenset(
        {"beauty:skincare", "beauty:makeup", "beauty:body_care", "beauty:hair", "beauty:"}
    ),
}
# 상품명에서만 찾는다. content 문서는 "향수 브랜드의 핸드크림", "와인 안주로 좋은 치즈"처럼
# 그 상품이 아닌 것을 함께 적는다. 카탈로그에 술 자체는 거의 없고 술잔·술 굿즈가 대부분이라
# 주류 제외는 술 관련 용품까지 뺀다.
_GIFT_EXCLUSION_TERMS: dict[GiftExclusion, tuple[str, ...]] = {
    GiftExclusion.PERFUME: (
        "향수",
        "오드퍼퓸",
        "오 드 퍼퓸",
        "오드뚜왈렛",
        "오 드 뚜왈렛",
        "오드트왈렛",
        "오드코롱",
        "오 드 코롱",
        "edp",
        "edt",
    ),
    GiftExclusion.ALCOHOL: (
        "와인",
        "맥주",
        "위스키",
        "소주",
        "막걸리",
        "사케",
        "샴페인",
        "하이볼",
        "리큐르",
        "전통주",
        "브랜디",
        "보드카",
        "데킬라",
        "칵테일",
        "약주",
        "청주",
        "과실주",
        "증류주",
        "술잔",
        "홈술",
        "혼술",
        "참이슬",
    ),
}


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def subcategory_key(category: str | None, subcategory: str | None) -> str:
    """구조 필터의 키. 세부 카테고리가 없으면 "food:"처럼 끝이 빈다."""
    return f"{category or ''}:{subcategory or ''}"


@dataclass(slots=True, frozen=True)
class CatalogExclusion:
    """추천 후보에서 반드시 빼는 상품의 조건."""

    categories: frozenset[str] = frozenset()
    # subcategory_key() 값
    subcategories: frozenset[str] = frozenset()
    # 모든 상품의 상품명에서 찾는 단어
    name_terms: tuple[str, ...] = ()
    # 먹거리 상품의 상품명과 content 문서에서 찾는 단어
    ingredient_terms: tuple[str, ...] = ()

    def blocks(self, product: CatalogProduct) -> bool:
        if product.category in self.categories:
            return True
        if subcategory_key(product.category, product.subcategory) in self.subcategories:
            return True
        name = _normalize(product.name)
        if any(term in name for term in self.name_terms):
            return True
        if self.ingredient_terms and _is_edible(product):
            text = f"{name} {_normalize(product.content_document or '')}"
            return any(term in text for term in self.ingredient_terms)
        return False


def _is_edible(product: CatalogProduct) -> bool:
    if product.category in _EDIBLE_CATEGORIES:
        return True
    return product.category == "voucher" and (
        product.subcategory is None or product.subcategory in _EDIBLE_VOUCHER_SUBCATEGORIES
    )


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_normalize(value) for value in values))


def build_catalog_exclusion(
    *,
    allergies: Iterable[Allergy] = (),
    gift_exclusions: Iterable[GiftExclusion] = (),
    exclude_pets: bool = False,
) -> CatalogExclusion:
    """exclude_pets 는 반려동물 신호가 없어 반려동물 대분류를 뺄 때 쓴다."""
    allergies = tuple(allergies)
    gift_exclusions = tuple(gift_exclusions)
    categories: set[str] = {PET_CATEGORY} if exclude_pets else set()
    subcategories: set[str] = set(_UNCLASSIFIED_EDIBLE) if allergies else set()
    for allergy in allergies:
        subcategories |= _ALLERGY_SUBCATEGORIES[allergy]
    for exclusion in gift_exclusions:
        categories |= _GIFT_EXCLUSION_CATEGORIES.get(exclusion, frozenset())
        subcategories |= _GIFT_EXCLUSION_SUBCATEGORIES.get(exclusion, frozenset())
    return CatalogExclusion(
        categories=frozenset(categories),
        # 대분류째 빠지는 세부 카테고리는 다시 적지 않는다.
        subcategories=frozenset(
            key for key in subcategories if key.split(":", 1)[0] not in categories
        ),
        name_terms=_unique(
            term
            for exclusion in gift_exclusions
            for term in _GIFT_EXCLUSION_TERMS.get(exclusion, ())
        ),
        ingredient_terms=_unique(term for allergy in allergies for term in _ALLERGY_TERMS[allergy]),
    )
