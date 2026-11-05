import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.domain.profile.models import ProfileSignal, TasteField
from app.domain.profile.taxonomy import (
    INTEREST_ROOT,
    TASTE_ROOT,
)

# 분석 화면 키워드. 저장 값을 묶고 다시 이름 붙인다. 결과 검증은 domain/profile/keywords.py 가 한다.
KEYWORD_SYSTEM = """니쥬가 한 사람과 이야기하며 모은 취향 항목과 관심사 항목을 보고,
그 사람을 소개하는 짧은 키워드를 고른다.
키워드는 본인과 친구에게 "취향", "관심사" 제목 아래 태그로 보인다.

[취향]
취향은 "무엇을 좋아하는가"에 대한 답이다. 대상이 아니라 그 사람이 좋아하는 결을 쓴다.
맛, 향, 색, 질감, 분위기, 디자인, 함께하는 사람, 즐기는 방식 같은 것이다.
- 대상 이름(target)은 떼고 결만 남긴다.
  ("포근한 향 바디로션" → "포근한 향", "산미 적고 고소한 커피" → "고소한 맛")
- 같은 결을 말하는 항목은 하나로 합치고 더 분명한 표현을 쓴다.
  ("향이 좋은 바디로션", "포근한 향 바디로션" → "포근한 향")
- 여러 대상에 같은 결이 보이면 그 공통점을 쓴다.
  ("혼자 가는 캠핑", "혼자 보는 영화" → "혼자 보내는 시간")
- 취향 키워드는 그 항목이 어떤 취향인지에 맞는 말로 쓴다.
  선호(무엇을 더 좋아하는지): "커피보다 차", "적당히 매운맛"
  감성(분위기, 감각, 미적 성향): "따뜻한 색감", "미니멀한 디자인", "포근한 향"
  스타일(개성, 패션, 디자인): "빈티지", "캐주얼"
  소비(가격대, 구매 성향): "실용적인 제품", "가성비"
  가치관(중요하게 보는 기준): "친환경", "튼튼함"
- "맛있는 떡볶이"처럼 대상 이름으로 끝나거나, "차 맛", "튼튼한 내구성"처럼
  같은 뜻을 겹쳐 쓰지 않는다.
- 위 예시 단어는 형태만 보여 주는 것이다. 입력 항목에 없는 결을 예시에서 가져와 쓰지 않는다.

[관심사]
관심사는 "현재 무엇에 관심을 가지고 있는가"에 대한 답이다.
시간과 마음을 쓰는 분야, 대상, 활동을 명사로 쓴다.
- 언제, 어떻게 하는지 묘사한 말은 그 대상이나 활동 이름으로 바꾼다.
  ("샤워 후 바디로션 바르기" → "바디케어", "퇴근 후 요리" → "요리")
- 비슷한 항목은 하나로 묶는다. ("디저트", "케이크 맛집" → "디저트 맛집")
- "음식", "운동"처럼 너무 넓은 이름으로 올리지 않고 사용자가 말한 수준에서 구체적으로 쓴다.

[고르는 법]
취향과 관심사를 각각 최대 3개 고른다. 앞에 있는 항목일수록 이야기가 많이 나온 것이다.
키워드는 2~10자의 자연스러운 한국어 명사구로 쓴다.
조사나 "~하기", "~좋아함" 같은 꼬리는 붙이지 않는다.
입력 항목이 가리키는 것만 쓰고, 입력에 없는 것을 짐작해서 만들지 않는다.
sources에는 그 키워드의 근거가 된 항목 id를 모두 넣는다.
취향은 tastes의 id, 관심사는 interests의 id다.
field와 axis는 분류 이름이라 그대로 키워드로 쓰지 않는다.
입력에 항목이 있는 칸은 비워 두지 않는다.

[예]
입력:
{"tastes":[{"id":0,"value":"포근한 향 바디로션","target":"바디로션"},
{"id":1,"value":"향이 좋은 바디로션","target":"바디로션"}],
"interests":[{"id":0,"value":"샤워 후 바디로션 바르기"},{"id":1,"value":"디저트"},
{"id":2,"value":"케이크 맛집"}]}
출력:
{"taste":[{"label":"포근한 향","sources":[0,1]}],
"interest":[{"label":"바디케어","sources":[0]},{"label":"디저트 맛집","sources":[1,2]}]}"""


class _KeywordModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KeywordPayload(_KeywordModel):
    label: str
    sources: list[int]


# 모델에 안내하는 출력 스키마. 실제 검증은 composed_keywords 가 키워드 단위로 한다.
# 두 칸을 필수로 두지 않으면 모델이 빈 결과로 답하기도 해서 필수로 둔다.
class KeywordSetPayload(_KeywordModel):
    taste: list[KeywordPayload] = Field(max_length=3)
    interest: list[KeywordPayload] = Field(max_length=3)


def build_keyword_messages(
    tastes: Sequence[ProfileSignal],
    interests: Sequence[ProfileSignal],
) -> list[dict[str, str]]:
    """id 는 각 목록 안의 위치다. 순위순으로 넘긴다."""

    def item(index: int, signal: ProfileSignal) -> dict[str, object]:
        path = signal.taxonomy_path
        label = "/".join(path[1:]) if path and path[0] in {TASTE_ROOT, INTEREST_ROOT} else None
        return {
            "id": index,
            "value": signal.value,
            **({"target": signal.target} if signal.target else {}),
            **(
                {"axis" if signal.field == TasteField.PREFERENCES else "field": label}
                if label
                else {}
            ),
        }

    return [
        {"role": "system", "content": KEYWORD_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "tastes": [item(index, signal) for index, signal in enumerate(tastes)],
                    "interests": [item(index, signal) for index, signal in enumerate(interests)],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
