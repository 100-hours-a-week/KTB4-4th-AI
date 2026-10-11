"""원본 상품의 product_category 를 상품 카테고리와 선물 테마로 나눈다.

원본 DB 의 product_category 는 카테고리 하나와 선물 테마 여러 개를 한 목록으로 담는다.
예: ["식품", "감사", "친구"]. 둘은 쓰임이 다르다.

- 카테고리(11개): 메인 추천(self)과 v2 카테고리 탭. "이 상품이 무엇인가"
- 선물 테마(6개): 마이 페이지와 친구 상세 페이지의 선물 목록(gift)에 나올 수 있는지
  (gift_available). 관계나 계기로 문서를 나누지 않는다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class ProductCategory(StrEnum):
    VOUCHER = "voucher"
    BEAUTY = "beauty"
    FOOD = "food"
    HEALTH = "health"
    LIVING = "living"
    FASHION = "fashion"
    DIGITAL = "digital"
    SPORTS = "sports"
    LUXURY = "luxury"
    BOOKS_TICKETS = "books_tickets"
    PET = "pet"


class GiftTheme(StrEnum):
    BIRTHDAY = "birthday"
    HOUSEWARMING = "housewarming"
    THANKS = "thanks"
    CHEER = "cheer"
    FRIEND = "friend"
    FUN = "fun"


class CategorySource(StrEnum):
    # 원본 product_category 에 있던 카테고리
    SOURCE = "source"
    # 원본에 카테고리가 없어 문서 생성 모델이 11개 중에서 골랐다.
    INFERRED = "inferred"


CATEGORY_LABELS: dict[ProductCategory, str] = {
    ProductCategory.VOUCHER: "교환권",
    ProductCategory.BEAUTY: "뷰티",
    ProductCategory.FOOD: "식품",
    ProductCategory.HEALTH: "건강",
    ProductCategory.LIVING: "리빙",
    ProductCategory.FASHION: "패션",
    ProductCategory.DIGITAL: "가전·디지털",
    ProductCategory.SPORTS: "레저·스포츠",
    ProductCategory.LUXURY: "명품·패션",
    ProductCategory.BOOKS_TICKETS: "책·음반티켓",
    ProductCategory.PET: "반려동물",
}

THEME_LABELS: dict[GiftTheme, str] = {
    GiftTheme.BIRTHDAY: "생일",
    GiftTheme.HOUSEWARMING: "이사/집들이",
    GiftTheme.THANKS: "감사",
    GiftTheme.CHEER: "응원",
    GiftTheme.FRIEND: "친구",
    GiftTheme.FUN: "재미",
}

# 테마가 없어도 선물 목록에 넣는 카테고리. 둘 다 받는 쪽보다 주는 쪽이 고르는 성격이 강하다.
GIFT_CATEGORIES = frozenset({ProductCategory.VOUCHER, ProductCategory.LUXURY})


def _normalize_label(label: str) -> str:
    # "명품 패션"/"명품패션", "이사/집들이"/"이사·집들이"처럼 표기가 조금씩 달라도 같은 라벨로 본다.
    return re.sub(r"[\s/·・]", "", label)


_CATEGORY_BY_LABEL = {
    _normalize_label(label): category for category, label in CATEGORY_LABELS.items()
}
_THEME_BY_LABEL = {_normalize_label(label): theme for theme, label in THEME_LABELS.items()}


@dataclass(slots=True, frozen=True)
class SourceLabels:
    categories: tuple[ProductCategory, ...]
    themes: tuple[GiftTheme, ...]
    unknown: tuple[str, ...]

    @property
    def category(self) -> ProductCategory | None:
        return self.categories[0] if self.categories else None


def split_labels(labels: Iterable[str]) -> SourceLabels:
    categories: list[ProductCategory] = []
    themes: list[GiftTheme] = []
    unknown: list[str] = []
    for label in labels:
        key = _normalize_label(label)
        if not key:
            continue
        if key in _CATEGORY_BY_LABEL:
            category = _CATEGORY_BY_LABEL[key]
            if category not in categories:
                categories.append(category)
        elif key in _THEME_BY_LABEL:
            theme = _THEME_BY_LABEL[key]
            if theme not in themes:
                themes.append(theme)
        else:
            unknown.append(label)
    return SourceLabels(
        categories=tuple(categories),
        # 저장과 해시가 원본의 라벨 순서에 흔들리지 않게 정의 순서로 둔다.
        themes=tuple(theme for theme in GiftTheme if theme in themes),
        unknown=tuple(unknown),
    )


def is_gift_available(category: ProductCategory | None, themes: Iterable[GiftTheme]) -> bool:
    """선물 목록(마이 페이지, 친구 상세 페이지)에 나올 수 있는지. 모든 상품은 메인 추천에 나온다."""
    return bool(tuple(themes)) or category in GIFT_CATEGORIES
