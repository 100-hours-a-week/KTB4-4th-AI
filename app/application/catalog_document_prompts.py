from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.application.ports.model_gateway import Message
from app.domain.catalog.models import (
    MAX_DOCUMENT_LENGTH,
    DocumentSourceProduct,
    PriceBand,
    price_band,
)

# 추천 질의는 app/domain/recommendation/queries.py 의 _query_text 가 만든다.
# 문서는 그 질의 문장과 같은 공간에서 가까워야 하므로,
# 아래 지시는 질의 템플릿을 그대로 따라간다.
#   content: "{취향값} 관련 제품" 또는 취향값 자체
#   usage:   "{취향값} 활동에서 사용하는 제품"
#   gift:    "선물하기 좋은 {취향값} 관련 제품"
SYSTEM_PROMPT = f"""너는 상품 검색용 임베딩 문서를 쓰는 작성자다.
주어진 상품을 content, usage, gift 세 가지 관점에서 각각 한국어 한두 문장으로 설명한다.

이 문서들은 사용자의 취향 문구와 의미 유사도를 비교하는 데 쓰인다. 취향 문구는
"캠핑", "홈카페", "러닝"처럼 짧은 명사구이거나 "캠핑 활동에서 사용하는 제품",
"선물하기 좋은 캠핑 관련 제품" 같은 형태다. 따라서 문서도 그 어휘와 맞물리게 쓴다.

## 세 문서의 역할

| space | 답하는 질문 | 쓰는 방식 |
| --- | --- | --- |
| content | 이 상품은 무엇인가 | 상품 종류, 소재, 형태, 용량 같은 속성을 명사 중심으로 쓴다 |
| usage | 누가 어떤 상황에서 쓰는가 | 사용하는 활동, 장소, 상황을 활동 이름으로 쓴다 |
| gift | 누구에게 왜 선물하기 좋은가 | 받는 사람의 관심사와 선물로 적합한 이유를 쓴다 |

## 규칙

1. 세 문서는 서로 달라야 한다. 같은 문장을 복사하거나 어순만 바꾸지 않는다.
2. 상품명에서 확인되는 사실만 쓴다. 브랜드 평판, 품질, 후기, 효능을 지어내지 않는다.
3. 상품명이 모호하면 일반적인 상위 범주로만 설명하고 세부 스펙을 추측하지 않는다.
4. 각 문서는 {MAX_DOCUMENT_LENGTH}자 이내로 쓴다.
5. 가격, 숫자 금액, 할인, URL, 상품 ID를 문서에 쓰지 않는다. 가격은 필터와 랭킹에서 따로 쓴다.
6. 광고 문구, 감탄사, 이모지, 마크다운, 따옴표를 쓰지 않는다.
7. usage 문서에는 그 상품이 쓰이는 활동 이름을 최소 하나 명시한다.
8. gift 문서에는 받는 사람을 "~하는 사람" 형태로 명시한다.
9. 상품명에 없는 계절, 기념일, 성별, 연령을 임의로 넣지 않는다.

## 가격 구간의 쓰임

가격 구간은 gift 문서의 서술 방향을 정할 때만 참고한다. 문서에 금액을 쓰지 않는다.

- low: 부담 없이 주고받는 선물, 여러 개를 함께 챙기기 좋은 물건으로 서술한다.
- mid: 취향을 아는 사람에게 건네는 일상 선물로 서술한다.
- high: 스스로 사기에는 부담스러워 미루게 되는 물건이라는 점을 서술한다.
- unknown: 가격 관점을 언급하지 않는다.

category_code 는 아래 목록에서 하나만 고른다. 확신이 없으면 other 를 쓴다.
digital, home, food, fashion, beauty, hobby, stationery, pet, other

판단할 수 없는 속성은 추측하지 말고 null 로 둔다."""

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "categoryCode",
        "categoryConfidence",
        "contentDocument",
        "usageDocument",
        "giftDocument",
        "attributes",
    ],
    "properties": {
        "categoryCode": {
            "type": "string",
            "enum": [
                "digital",
                "home",
                "food",
                "fashion",
                "beauty",
                "hobby",
                "stationery",
                "pet",
                "other",
            ],
        },
        "categoryConfidence": {"type": "number", "minimum": 0, "maximum": 1},
        "contentDocument": {"type": "string", "maxLength": MAX_DOCUMENT_LENGTH},
        "usageDocument": {"type": "string", "maxLength": MAX_DOCUMENT_LENGTH},
        "giftDocument": {"type": "string", "maxLength": MAX_DOCUMENT_LENGTH},
        "attributes": {
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
        },
    },
}

_EXAMPLE_USER = json.dumps(
    {"productName": "티타늄 더블월 머그컵 350ml", "priceBand": "mid"},
    ensure_ascii=False,
)
_EXAMPLE_ASSISTANT = json.dumps(
    {
        "categoryCode": "hobby",
        "categoryConfidence": 0.9,
        "contentDocument": (
            "티타늄 소재의 350ml 더블월 머그컵. 보온이 되는 아웃도어용 컵과 캠핑 식기."
        ),
        "usageDocument": (
            "캠핑이나 등산, 야외 커피처럼 밖에서 따뜻한 음료를 마시는 활동에서 쓰는 머그컵."
        ),
        "giftDocument": "캠핑과 야외 커피를 즐기는 사람에게 건네기 좋은 휴대용 티타늄 머그컵.",
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
    user_payload = json.dumps(
        {
            "productName": product.name,
            "priceBand": _band_label(price_band(product.price)),
        },
        ensure_ascii=False,
    )
    return (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _EXAMPLE_USER},
        {"role": "assistant", "content": _EXAMPLE_ASSISTANT},
        {"role": "user", "content": user_payload},
    )
