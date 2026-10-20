from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.application.ports.model_gateway import Message
from app.domain.catalog.labels import CATEGORY_LABELS, GiftTheme
from app.domain.catalog.models import (
    MAX_DOCUMENT_LENGTH,
    DocumentSourceProduct,
    PriceBand,
    price_band,
)

# 추천 질의는 app/domain/recommendation/queries.py 의 _query_text 가 만든다.
# 관심사별 검색 의도(app/application/search_intent_prompts.py)가 있으면 의도 문장을 쓴다.
# 의도 문장은 이 문서 형식("{카테고리} {세부 카테고리}. ...")을 따라 쓰므로, 이 형식을 바꾸면
# 의도 프롬프트의 예시와 INTENT_VERSION 도 함께 바꾼다.
# 의도가 없는 관심사는 아래 템플릿으로 찾으므로, 문서도 템플릿과 같은 어휘를 쓴다.
#   content: "{취향값} 관련 제품" 또는 취향값 자체
#   usage:   "{취향값} 활동에서 사용하는 제품"
#   gift:    "{취향값}에 관심 있는 사람이 받으면 좋은 선물"
SYSTEM_PROMPT = f"""너는 선물 추천 서비스의 상품 검색용 임베딩 문서를 쓰는 작성자다.
주어진 상품을 정해진 관점마다 한국어 한두 문장으로 설명한다.

이 문서들은 사용자의 취향·관심사 문구와 의미 유사도를 비교하는 데 쓰인다. 취향 문구는
"캠핑", "홈카페", "러닝"처럼 짧은 명사구이거나 "캠핑 활동에서 사용하는 제품",
"캠핑에 관심 있는 사람이 받으면 좋은 선물" 같은 형태다. 문서도 그 어휘와 맞물리게 쓴다.

## 입력

- productName: 판매 문구를 걷어낸 상품명. 상품 사실은 여기서만 가져온다.
- category: 상품 카테고리. null 이면 아래 목록에서 직접 고른다.
- subcategory: 카테고리 안에서 이 상품이 어떤 종류인지. null 이면 productName 으로 판단한다.
- priceBand: 가격 구간. gift 문서의 서술 방향에만 참고한다.
- writeGift: true 면 gift 문서를 쓴다. false 면 gift 문서를 쓰지 않는다.
- giftHints: gift 문서에 녹일 수 있는 선물 성격. 비어 있으면 쓰지 않는다.

## 문서의 역할

- content: 이 상품은 무엇인가. 상품 종류, 브랜드, 이름에 나온 속성을 명사 중심으로 쓴다.
- usage: 언제, 어디서, 어떤 활동에 쓰는가. 사용하는 활동, 장소, 상황을 활동 이름으로 쓴다.
- gift: 어떤 취향·관심사를 가진 사람이 받으면 좋은가. 받는 사람을 "~을 좋아하는 사람",
  "~에 관심 있는 사람"으로 쓰고 선물로서의 이유를 붙인다.

## subcategory 쓰는 법

- subcategory 는 상품 종류를 미리 판단해 둔 값이다. 상품 종류는 이 값을 기준으로 쓴다.
  productName 이 분명히 다른 종류를 가리킬 때만 productName 을 따른다.
- content 문서는 "{{category}} {{subcategory}}." 로 시작한다. 예: "레저·스포츠 캠핑 용품."
  subcategory 가 null 이면 "{{category}}" 다음에 productName 에서 확인되는 상품 종류를 쓴다.
- subcategory 에 활동 이름이 들어 있으면(캠핑 용품, 러닝 용품, 골프 용품, 홈트·요가 용품)
  usage 문서에 그 활동을 쓴다.
- 아래 종류는 따로 지킨다.
  - 향수, 디퓨저·방향제: 이름에 있는 향 계열만 쓴다. 이름에 없으면 향 계열을 쓰지 않는다.
  - 도서: 제목만으로 줄거리나 주제를 지어내지 않는다. 장르를 모르면 "도서"라고만 쓴다.
  - 금액권, 디지털 이용권: 쓸 수 있는 매장이나 서비스가 이름에 있으면 쓴다.
    없으면 "모바일 금액권"으로 쓴다.
  - 반려동물 종류: 강아지용인지 고양이용인지 이름에 있으면 쓴다.

## 카테고리별로 꼭 쓸 것과 쓰지 말 것

| 카테고리 | content 에 꼭 쓸 것 | 쓰지 말 것 |
| --- | --- | --- |
| 교환권 | 브랜드와 무엇으로 바꾸는지(커피, 디저트, 치킨, 편의점 간식, 금액권) | 금액 |
| 뷰티 | 제품 종류, 이름에 있는 향 계열 | 효능, 피부 개선 주장 |
| 식품 | 음식 종류, 먹는 상황 | 맛 평가 |
| 건강 | 제품 종류 | 효능, 질병 개선 주장 |
| 리빙 | 제품 종류, 놓이는 공간 | 성능 |
| 패션 | 아이템 종류, 이름에 있는 스타일 | 소재 지어내기 |
| 가전·디지털 | 제품 종류, 용도 | 성능 수치 지어내기 |
| 레저·스포츠 | 종목이나 활동 이름(캠핑, 러닝, 골프 등) | 종목을 모르는데 지어내기 |
| 명품·패션 | 브랜드, 아이템 종류 | 소재 지어내기 |
| 책·음반티켓 | 장르, 공연·전시·음반 종류 | 내용 요약 지어내기 |
| 반려동물 | 강아지용인지 고양이용인지 | 받는 대상을 사람처럼 쓰기 |

## 규칙

1. 문서끼리 서로 달라야 한다. 같은 문장을 복사하거나 어순만 바꾸지 않는다.
2. content 문서의 상품 사실은 productName 에서 확인되는 것만 쓴다.
   질감, 향의 강도, 성능, 품질, 후기, 효능을 지어내지 않는다.
3. usage 와 gift 문서에서는 상품 종류에 맞는 사용 상황과 받는 사람의 취향을 추론해도 된다.
   단, 추론한 내용을 상품 자체의 스펙이나 성능처럼 쓰지 않는다.
4. 각 문서는 {MAX_DOCUMENT_LENGTH}자 이내로 쓴다.
5. 가격, 숫자 금액, 할인, URL, 상품 ID를 쓰지 않는다. 가격은 필터와 랭킹에서 따로 쓴다.
6. 광고 문구, 감탄사, 이모지, 마크다운, 따옴표를 쓰지 않는다.
7. usage 문서에는 그 상품이 쓰이는 활동이나 상황 이름을 최소 하나 쓴다.
8. gift 문서에는 받는 사람을 취향·관심사로 쓴다. 관계나 성별로 쓰지 않는다.
   "남자친구", "여자친구", "연인", "부모님", "남자에게", "여성에게" 같은 표현을 쓰지 않는다.
9. gift 문서에 생일, 감사, 응원 같은 주는 계기를 쓰지 않는다. 계기는 필터에서 따로 쓴다.
10. productName 에 없는 계절, 기념일, 성별, 연령을 임의로 넣지 않는다.
11. productName 에 직접 없는 숫자, 용량, 개수, 소재, 구성품, 성능을 만들지 않는다.
12. "프리미엄", "고급형", "품질이 좋다"처럼 확인할 수 없는 평가를 쓰지 않는다.

## giftHints

- housewarming: 새집이나 혼자 사는 공간에 두기 좋다는 점을 녹일 수 있다.
- fun: 받는 사람에게 웃음을 주는 재미있는 선물이라는 점을 녹일 수 있다.
- cheer: 바쁘거나 지친 일상에 힘이 되고 기분 전환이 된다는 점을 녹일 수 있다.
  상품마다 같은 문장을 붙이지 말고 상품 종류에 맞게 쓴다.

## 가격 구간의 쓰임

- low: 부담 없이 주고받는 선물로 서술한다.
- mid: 취향을 아는 사람에게 건네는 일상 선물로 서술한다.
- high: 스스로 사기에는 부담스러워 미루게 되는 물건이라는 점을 서술한다.
- unknown: 가격 관점을 언급하지 않는다.

## category 가 null 일 때

아래 11개 중 하나를 category 로 고르고, 확신 정도를 categoryConfidence(0~1)로 쓴다.
교환권, 뷰티, 식품, 건강, 리빙, 패션, 가전·디지털, 레저·스포츠, 명품·패션, 책·음반티켓,
반려동물

- 교환권: 매장에서 바꾸거나 배달받는 모바일 쿠폰. 카페 음료, 편의점 상품, 치킨·버거 세트,
  매장 케이크, 금액권과 충전권이 여기 들어간다.
- 꽃, 꽃다발, 화분, 식물은 리빙이다.
- 골드바와 순금 제품은 명품·패션이다.

## attributes 작성 기준

- 알 수 없는 값은 추측하지 말고 null 로 쓴다.
- needsSize 는 의류·신발처럼 구매자가 규격을 골라야 하는 상품만 true 로 쓴다.
- needsInstall 은 설치·조립이 명시되거나 설치가 필수인 상품만 true 로 쓴다.
- materials 에는 productName 에 직접 등장한 소재만 넣는다. 없으면 빈 배열로 쓴다.
- hasScent 와 containsAlcohol 은 productName 에서 직접 확인될 때만 true 로 쓴다."""

# 받는 사람의 상황·취향으로 풀어 쓸 수 있는 테마만 힌트로 넘긴다.
# 생일·감사·친구는 주는 사람의 계기나 관계라서 문서에 넣지 않는다.
_GIFT_HINTS = {
    GiftTheme.HOUSEWARMING: "housewarming",
    GiftTheme.CHEER: "cheer",
    GiftTheme.FUN: "fun",
}

_ATTRIBUTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "isConsumable",
        "needsSize",
        "hasScent",
        "needsInstall",
        "containsAlcohol",
        "materials",
    ],
    "properties": {
        "isConsumable": {"type": ["boolean", "null"]},
        "needsSize": {"type": ["boolean", "null"]},
        "hasScent": {"type": ["boolean", "null"]},
        "needsInstall": {"type": ["boolean", "null"]},
        "containsAlcohol": {"type": ["boolean", "null"]},
        "materials": {"type": "array", "items": {"type": "string"}},
    },
}


def build_json_schema(product: DocumentSourceProduct) -> dict[str, Any]:
    """상품마다 스키마를 만든다. 쓰지 않을 필드는 아예 요구하지 않는다.

    gift 문서는 gift_available 상품에만, 카테고리 선택은 카테고리를 모르는 상품에만 요구한다.
    """
    document = {"type": "string", "maxLength": MAX_DOCUMENT_LENGTH}
    properties: dict[str, Any] = {
        "contentDocument": document,
        "usageDocument": document,
        "attributes": _ATTRIBUTES_SCHEMA,
    }
    required = ["contentDocument", "usageDocument", "attributes"]
    if product.gift_available:
        properties["giftDocument"] = document
        required.append("giftDocument")
    if product.category is None:
        properties["category"] = {"type": "string", "enum": list(CATEGORY_LABELS.values())}
        properties["categoryConfidence"] = {"type": "number", "minimum": 0, "maximum": 1}
        required.extend(["category", "categoryConfidence"])
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


_EXAMPLE_USER = json.dumps(
    {
        "productName": "티타늄 더블월 머그컵 350ml",
        "category": "레저·스포츠",
        "subcategory": "캠핑 용품",
        "priceBand": "mid",
        "writeGift": True,
        "giftHints": [],
    },
    ensure_ascii=False,
)
_EXAMPLE_ASSISTANT = json.dumps(
    {
        "contentDocument": "레저·스포츠 캠핑 용품. 티타늄 소재의 350ml 더블월 구조 머그컵.",
        "usageDocument": (
            "캠핑이나 등산, 야외 커피처럼 밖에서 따뜻한 음료를 마시는 활동에서 쓰는 머그컵."
        ),
        "giftDocument": (
            "캠핑과 야외 커피를 좋아하는 사람이 받으면 좋은 휴대용 머그컵. "
            "캠핑을 좋아하는 사람에게 건네기 좋은 일상 선물."
        ),
        "attributes": {
            "isConsumable": False,
            "needsSize": False,
            "hasScent": False,
            "needsInstall": False,
            "containsAlcohol": False,
            "materials": ["티타늄"],
        },
    },
    ensure_ascii=False,
)


def _band_label(band: PriceBand | None) -> str:
    return band.value if band is not None else "unknown"


def build_document_messages(product: DocumentSourceProduct) -> Sequence[Message]:
    gift_available = product.gift_available
    user_payload = json.dumps(
        {
            "productName": product.name,
            "category": (
                CATEGORY_LABELS[product.category] if product.category is not None else None
            ),
            "subcategory": product.subcategory_label,
            "priceBand": _band_label(price_band(product.price)),
            "writeGift": gift_available,
            "giftHints": (
                [_GIFT_HINTS[theme] for theme in product.themes if theme in _GIFT_HINTS]
                if gift_available
                else []
            ),
        },
        ensure_ascii=False,
    )
    return (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _EXAMPLE_USER},
        {"role": "assistant", "content": _EXAMPLE_ASSISTANT},
        {"role": "user", "content": user_payload},
    )
