from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.application.ports.model_gateway import Message
from app.domain.recommendation.reasons import ReasonClues

# 한 번에 다시 쓰는 상품 수. 목록을 나눠 동시에 부르면 한 번에 길게 쓰는 것보다 빨리 끝난다.
REASON_CHUNK_SIZE = 5
MIN_REASON_LENGTH = 10
MAX_REASON_LENGTH = 80

REASON_SYSTEM = """너는 선물 추천 서비스 니쥬의 상품 목록 옆에 붙는 추천 이유를 쓴다.

[목록]
{audience}

[입력]
items의 각 상품에 id, product(상품 이름), category, likes, new, hesitant가 있다.
- likes: 이 사람이 좋아하는 것 중 이 상품과 이어지는 것이다. 앞에 있을수록 이 상품과 가깝다.
  kind는 관심사, 즐겨 하는 것, 취향, 갖고 싶어 한 것, 자주 쓰는 것 중 하나다.
- new: true면 이 사람의 관심사에서 한 걸음 넓혀서 찾은 상품이다.
- hesitant: true면 스스로 사기엔 망설여질 만한 종류의 상품이다.

[쓰는 법]
상품마다 한 문장으로, 20~45자 정도로 쓴다.
이 상품이 likes와 어떻게 이어지는지를 이 상품에 맞게 쓴다.
likes 값을 문장에 그대로 끼워 넣지 말고 자연스러운 말로 바꿔서 녹인다.
("샤워 후 바디로션 바르기" → "샤워하고 바디케어를 꼭 챙기는",
"향이 좋은 바디로션" → "좋은 향을 아끼는")
likes는 이 상품과 이어지는 것만 한두 개 쓰고, 이어지지 않으면 빼도 된다.
new가 true면 "~을 좋아한다면 이런 것도 반가울 거예요"처럼
관심사에서 넓혀 본 추천이라는 게 드러나게 쓴다.
hesitant가 true면 무엇을 못 샀는지는 말하지 않고, 선물로 받으면 반가울 거라는 쪽으로만 쓴다.
목록 안에서 같은 문형이 이어지지 않게 표현을 바꾼다. 다만 어법이 어색해지면 안 된다.

[하지 않는 것]
상품 이름에 없는 성능, 소재, 효과, 후기를 지어내지 않는다. 브랜드 이름은 쓰지 않는다.
"취향을 반영했어요", "관심과 잘 맞는 상품이에요"처럼 어떤 상품에나 붙는 말로 끝내지 않는다.
담백한 존댓말("~예요", "~이에요", "~거예요")로 쓰고 이모지와 느낌표는 쓰지 않는다.
상품마다 이유를 하나씩 빠짐없이 쓴다."""

_SELF_AUDIENCE = (
    "이 목록은 본인이 보는 추천 목록이다. 사람을 부르지 않고 이 상품이 왜 맞는지 말한다."
)
_GIFT_AUDIENCE = (
    "이 목록은 친구가 이 사람에게 줄 선물을 고르며 보는 목록이다. "
    '받는 사람은 "{who}"이라고 부르고, 다른 호칭은 쓰지 않는다. '
    "선물하는 친구에게 무엇을 하라고 시키지 않는다."
)


class _ReasonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReasonPayload(_ReasonModel):
    id: int
    reason: str


# 모델에 안내하는 출력 스키마. 실제 검증은 상품 단위로 한다.
# 필수로 두지 않으면 모델이 빈 결과로 답하기도 해서 필수로 둔다.
class ReasonSetPayload(_ReasonModel):
    reasons: list[ReasonPayload] = Field(max_length=REASON_CHUNK_SIZE)


def reason_system(*, gift: bool, recipient_name: str | None) -> str:
    """recipient_name 은 username 원문이다. 선물 목록의 받는 사람은 원문에 "님"을 붙인다."""
    if not gift:
        return REASON_SYSTEM.format(audience=_SELF_AUDIENCE)
    who = f"{recipient_name}님" if recipient_name else "받는 분"
    return REASON_SYSTEM.format(audience=_GIFT_AUDIENCE.format(who=who))


def build_reason_messages(
    items: Sequence[tuple[str, str | None, ReasonClues]],
    *,
    gift: bool,
    recipient_name: str | None,
) -> list[Message]:
    """items 는 (상품 이름, 카테고리, 근거)이고 id 는 그 위치다."""
    return [
        {"role": "system", "content": reason_system(gift=gift, recipient_name=recipient_name)},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "items": [
                        {
                            "id": index,
                            "product": name,
                            **({"category": category} if category else {}),
                            "likes": [
                                {"value": like.value, "kind": like.kind} for like in clues.likes
                            ],
                            "new": clues.exploration,
                            "hesitant": clues.hesitant,
                        }
                        for index, (name, category, clues) in enumerate(items)
                    ]
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
