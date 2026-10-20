from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.application.ports.model_gateway import Message
from app.domain.catalog.labels import CATEGORY_LABELS
from app.domain.catalog.subcategories import SUBCATEGORIES
from app.domain.recommendation.intents import (
    MAX_PRODUCT_TYPE_LENGTH,
    MAX_PRODUCT_TYPES,
    MAX_QUERY_LENGTH,
    MAX_SENSE_LENGTH,
    MIN_PRODUCT_TYPES,
    InvalidSearchIntent,
)

# 상품 content 문서는 "{대분류} {세부 카테고리}. ..."로 시작한다(catalog_document_prompts.py).
# 쿼리도 같은 머리말로 시작해야 가까워지는데, 머리말이 틀리면 크게 멀어진다.
# ("식품 차·커피·음료. 와인잔"은 와인잔 대신 허브티를, "리빙 주방·식기. 와인잔"은 와인잔을 찾았다)
# 그래서 모델은 이 목록에서 고르기만 하고 문장 앞에는 코드가 붙인다.
CATALOG_CATEGORIES: tuple[str, ...] = tuple(
    f"{label} {subcategory}"
    for category, label in CATEGORY_LABELS.items()
    for subcategory in SUBCATEGORIES[category].values()
)
# 머리말을 붙인 뒤에도 MAX_QUERY_LENGTH 안에 들어가도록 모델이 쓰는 꼬리 문장 길이를 줄인다.
_TAIL_LENGTH = MAX_QUERY_LENGTH - max(len(category) for category in CATALOG_CATEGORIES) - 2

SEARCH_INTENT_SYSTEM = f"""너는 선물 추천 서비스의 상품 검색 문장을 쓰는 작성자다.
사용자의 관심사 하나를 받아, 상품을 찾을 문장을 관점마다 하나씩 쓴다.

이 문장은 상품 문서와 의미 유사도를 비교하는 데 쓰인다. 상품 문서는 아래처럼 쓰여 있다.
- content: "레저·스포츠 캠핑 용품. 티타늄 소재의 350ml 더블월 구조 머그컵."
- usage: "캠핑이나 등산, 야외 커피처럼 밖에서 따뜻한 음료를 마시는 활동에서 쓰는 머그컵."
- gift: "캠핑과 야외 커피를 좋아하는 사람이 받으면 좋은 휴대용 머그컵."
네 문장도 이 문형과 어휘에 맞물리게 쓴다.

## 입력
- value: 사용자의 관심사. 사용자 표현 그대로다.
- field: 관심사 분야. null 이면 value 만 보고 정한다.

## 상품으로 쓰지 않는 것
술(와인, 맥주, 위스키), 담배, 의약품, 건강식품, 살아 있는 동물, 서비스(강습, 여행 상품), 브랜드는
상품 종류로 쓰지 않는다. 관심사가 이런 것이면 그것과 함께 쓰는 물건을 쓴다.
(와인 → 와인잔, 와인 오프너, 디캔터)

## 쓰는 순서
1. sense: value 가 무슨 뜻인지 {MAX_SENSE_LENGTH}자 이내로 정한다.
   뜻이 여럿이면 field 로 고른다. ("차"가 음식·미식이면 마시는 차, 자동차면 탈 것)
2. productTypes: 이 관심사가 있는 사람이 쓰거나 받으면 좋은, 선물 가게에서 파는 상품 종류를
   {MIN_PRODUCT_TYPES}~{MAX_PRODUCT_TYPES}개 쓴다. 각 {MAX_PRODUCT_TYPE_LENGTH}자 이내의
   구체적인 종류 이름이다(방한 장갑, 넥워머).
   관심사 자체가 선물 가게에서 파는 상품이면(게임 타이틀, 책, 원두, 향수) 그 상품을 첫 자리에
   쓰고, 함께 쓰는 물건은 뒤에 덧붙인다. (콘솔게임 → 게임 타이틀, 컨트롤러)
3. category: productTypes 의 첫 상품이 실제로 진열될 카테고리를 아래 목록에서 그대로 하나 고른다.
   (와인잔은 "리빙 주방·식기", 원두는 "식품 차·커피·음료")
4. content: 카테고리 이름 없이, 그 관심사에서 쓰는 productTypes 를 한 문장으로 쓴다.
   예: "서핑할 때 쓰는 방수 파우치, 비치 타월, 래시가드."
5. usage: 그 관심사의 활동, 장소, 상황 이름과 productTypes 를 함께 쓴다.
6. gift: "~을 좋아하는 사람이 받으면 좋은 ~" 문형으로 관심사와 productTypes 를 쓴다.
7. explore: 카탈로그에 이 관심사의 상품이 없을 때 대신 제안할 상품을 찾는 문장이다.
   이 관심사를 좋아하는 사람이 그 활동에서 무엇을 즐기는지 한 가지로 정하고, 그 즐거움과 이어지는
   상품을 productTypes 와 다른 종류로 2~4개 골라 카테고리 이름 없이 한 문장으로 쓴다.
   그 관심사를 하는 상황이나 같은 즐거움 안에 머문다. 피로, 건강, 수면처럼 몸 상태로 넓히지 않는다.
   예: 볼링 → "친구들과 함께하는 보드게임과 실내 놀이."
8. exploreCategory: explore 상품이 진열될 카테고리를 아래 목록에서 그대로 하나 고른다.

## 규칙
- value 에 없는 대상을 끌어오지 않는다. 사람이 하는 활동이면 반려동물 용품을 쓰지 않는다.
  ("산책"은 걷기 좋은 운동화나 텀블러다. 강아지 목줄이 아니다)
  반려동물 용품은 value 나 field 가 반려동물일 때만 쓴다.
- 글자가 비슷한 다른 단어로 쓰지 않는다. ("스키"에 스키니, 스킨케어를 쓰지 않는다)
- 카탈로그에 그 관심사 전용 상품이 없을 수 있다. 그때는 그 관심사를 즐길 때 함께 쓰는
  일반 상품으로 쓴다. (서핑 → 방수 파우치, 비치 타월)
- 관계, 성별, 나이, 가격, 기념일은 쓰지 않는다.
- content 와 explore 는 {_TAIL_LENGTH}자, usage 와 gift 는 {MAX_QUERY_LENGTH}자 이내의 한국어
  문장이다. 마크다운, 따옴표, 이모지는 쓰지 않는다.

## 카테고리 목록
{chr(10).join(f"- {category}" for category in CATALOG_CATEGORIES)}"""

SEARCH_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "sense",
        "productTypes",
        "category",
        "content",
        "usage",
        "gift",
        "explore",
        "exploreCategory",
    ],
    "properties": {
        "sense": {"type": "string", "maxLength": MAX_SENSE_LENGTH},
        "productTypes": {
            "type": "array",
            "minItems": MIN_PRODUCT_TYPES,
            "maxItems": MAX_PRODUCT_TYPES,
            "items": {"type": "string", "maxLength": MAX_PRODUCT_TYPE_LENGTH},
        },
        "category": {"type": "string", "enum": list(CATALOG_CATEGORIES)},
        "content": {"type": "string", "maxLength": _TAIL_LENGTH},
        "usage": {"type": "string", "maxLength": MAX_QUERY_LENGTH},
        "gift": {"type": "string", "maxLength": MAX_QUERY_LENGTH},
        "explore": {"type": "string", "maxLength": _TAIL_LENGTH},
        "exploreCategory": {"type": "string", "enum": list(CATALOG_CATEGORIES)},
    },
}

# 예시는 카탈로그에 전용 상품이 없는 관심사로 둔다. 추천이 자주 실패하던 경우다.
_EXAMPLE_USER = json.dumps({"value": "서핑", "field": "아웃도어"}, ensure_ascii=False)
_EXAMPLE_ASSISTANT = json.dumps(
    {
        "sense": "파도를 타는 해양 스포츠 서핑",
        "productTypes": ["방수 파우치", "비치 타월", "래시가드", "선스틱"],
        "category": "레저·스포츠 스포츠·아웃도어 용품",
        "content": "서핑할 때 쓰는 방수 파우치, 비치 타월, 래시가드.",
        "usage": "바다에서 서핑을 하고 물놀이를 할 때 쓰는 방수 파우치, 비치 타월, 선스틱.",
        "gift": "서핑을 좋아하는 사람이 받으면 좋은 방수 파우치와 비치 타월, 래시가드.",
        "explore": "바다 분위기를 집에서도 즐기는 무드등과 바다 향 디퓨저.",
        "exploreCategory": "리빙 조명·인테리어 소품",
    },
    ensure_ascii=False,
)


def build_search_intent_messages(value: str, field: str | None) -> Sequence[Message]:
    return (
        {"role": "system", "content": SEARCH_INTENT_SYSTEM},
        {"role": "user", "content": _EXAMPLE_USER},
        {"role": "assistant", "content": _EXAMPLE_ASSISTANT},
        {
            "role": "user",
            "content": json.dumps({"value": value, "field": field}, ensure_ascii=False),
        },
    )


def intent_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """모델 응답의 카테고리를 content·explore 문장 앞에 붙여 SearchIntent 형식으로 바꾼다."""

    def headed(category_key: str, text_key: str) -> str:
        category = raw.get(category_key)
        if category not in CATALOG_CATEGORIES:
            raise InvalidSearchIntent(f"{category_key} must be one of the catalog categories")
        text = raw.get(text_key)
        if not isinstance(text, str) or not text.strip():
            raise InvalidSearchIntent(f"{text_key} must be a non-empty string")
        return f"{category}. {text.strip()}"

    return {
        "sense": raw.get("sense"),
        "productTypes": raw.get("productTypes"),
        "content": headed("category", "content"),
        "usage": raw.get("usage"),
        "gift": raw.get("gift"),
        "explore": headed("exploreCategory", "explore"),
    }
