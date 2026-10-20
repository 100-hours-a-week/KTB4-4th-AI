"""관심사 신호마다 미리 만들어 두는 검색 의도.

"스키" 같은 값 하나를 템플릿에 넣으면 쿼리가 단어 하나만 담는다. 맞는 상품이 카탈로그에 없으면
임베딩은 글자 모양이 비슷한 상품(스키니 스카프, 스킨케어)을 가장 가깝다고 돌려준다.
검색 의도는 그 관심사가 어떤 뜻인지 정하고, 어떤 상품 종류와 이어지는지를 상품 문서와 같은
문형으로 공간마다 풀어 쓴다.

- content: 무엇인가. 상품 content 문서처럼 "{카테고리} {상품 종류}."로 시작한다.
- usage: 어떤 활동과 상황에서 쓰는가.
- gift: 그 관심사가 있는 사람이 받으면 좋은 것.
- explore: 그 관심사로 즐기는 것을 다른 종류의 상품으로 가리킨다. 카탈로그에 그 관심사의 상품이
  없을 때 탐색 추천을 찾는 데 쓴다. ("볼링" → 친구들과 함께하는 보드게임과 실내 놀이)

의도는 관심사 값과 분야만으로 정해지므로 사용자끼리 같이 쓴다. 사용자마다 다른 맥락(상황 취향,
생활 맥락)은 의도에 넣지 않고 추천할 때 따로 붙인다.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.domain.recommendation.models import VectorSpace

# 의도 생성 프롬프트나 스키마를 바꾸면 올린다. 저장소 키에 들어가서 예전 의도를 쓰지 않는다.
INTENT_VERSION = "v2"
MAX_SENSE_LENGTH = 40
MAX_QUERY_LENGTH = 160
MIN_PRODUCT_TYPES = 2
MAX_PRODUCT_TYPES = 5
MAX_PRODUCT_TYPE_LENGTH = 20


class InvalidSearchIntent(ValueError):
    """모델이 만든 의도가 형식에 맞지 않을 때."""


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def intent_key(value: str, taxonomy_path: Sequence[str] | None) -> str:
    """같은 관심사를 사용자끼리 같이 쓰는 키. 분야가 다르면 뜻이 다를 수 있어 키에 넣는다.

    "차"는 음식·미식 분야면 마시는 차, 자동차 분야면 탈 것이다.
    """
    path = "/".join(_normalize(part) for part in taxonomy_path or ())
    return f"{_normalize(value)}|{path}"


def interest_field_name(taxonomy_path: Sequence[str] | None) -> str | None:
    """("관심사", "아웃도어") 같은 분야 경로에서 프롬프트에 넣을 분야 이름."""
    if not taxonomy_path or len(taxonomy_path) < 2:
        return None
    return str(taxonomy_path[-1])


def _text(payload: Mapping[str, Any], key: str, limit: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidSearchIntent(f"{key} must be a non-empty string")
    text = " ".join(value.split())
    if len(text) > limit:
        raise InvalidSearchIntent(f"{key} must be at most {limit} characters")
    return text


@dataclass(slots=True, frozen=True)
class SearchIntent:
    sense: str
    product_types: tuple[str, ...]
    content: str
    usage: str
    gift: str
    explore: str

    def text(self, space: VectorSpace) -> str:
        if space == VectorSpace.USAGE:
            return self.usage
        if space == VectorSpace.GIFT:
            return self.gift
        return self.content

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SearchIntent:
        """모델 응답이나 저장소 값을 검증해서 만든다. 형식이 틀리면 InvalidSearchIntent."""
        types = payload.get("productTypes")
        if not isinstance(types, Sequence) or isinstance(types, (str, bytes)):
            raise InvalidSearchIntent("productTypes must be an array")
        product_types = tuple(
            dict.fromkeys(" ".join(str(item).split()) for item in types if str(item).strip())
        )
        if not MIN_PRODUCT_TYPES <= len(product_types) <= MAX_PRODUCT_TYPES:
            raise InvalidSearchIntent(
                f"productTypes must have {MIN_PRODUCT_TYPES}-{MAX_PRODUCT_TYPES} items"
            )
        if any(len(item) > MAX_PRODUCT_TYPE_LENGTH for item in product_types):
            raise InvalidSearchIntent("productTypes items are too long")
        return cls(
            sense=_text(payload, "sense", MAX_SENSE_LENGTH),
            product_types=product_types,
            content=_text(payload, "content", MAX_QUERY_LENGTH),
            usage=_text(payload, "usage", MAX_QUERY_LENGTH),
            gift=_text(payload, "gift", MAX_QUERY_LENGTH),
            explore=_text(payload, "explore", MAX_QUERY_LENGTH),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "sense": self.sense,
            "productTypes": list(self.product_types),
            "content": self.content,
            "usage": self.usage,
            "gift": self.gift,
            "explore": self.explore,
        }
