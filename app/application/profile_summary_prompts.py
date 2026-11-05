import json

from app.domain.profile.merger import friend_signal_groups, taste_rank_score
from app.domain.profile.models import ProfileState, TasteField
from app.domain.profile.taxonomy import (
    INTEREST_ROOT,
    TASTE_ROOT,
)

# 요약문에 넘기는 단서 수. 요약 타임아웃이 짧아서 입력이 지나치게 길어지지 않게 한다.
SUMMARY_CLUE_LIMIT = 20
# 요약에 대비로 드러낼 별로라고 한 것의 수.
SUMMARY_AVOID_LIMIT = 3

_SUMMARY_FIELD_LABELS = {
    TasteField.HOBBIES: "자주 하는 활동",
    TasteField.INTERESTS: "관심을 보인 것",
    TasteField.PREFERENCES: "좋아하는 방식과 스타일",
}

_FRIEND_SUMMARY_TEMPLATE = """너는 선물 추천 서비스 니쥬다.
니쥬가 이 사람과 이야기해 보고 알게 된 취향과 관심사를
다른 사람에게 "{topic} 이런 사람이에요" 하고 소개하는 글을 쓴다.
이 글은 본인에게 먼저 보여 주고, 본인이 허락하면 친구에게도 보여 준다.
읽는 사람이 이 사람이 무엇을 좋아하고 어떤 데 관심이 있는지 그려질 수 있게 쓴다.
글 안에서 읽는 사람을 부르지 않고, "~해 주면 좋아할 것 같아요"처럼 무엇을 해 주라고 제안하지 않는다.

[쓰는 방법]
분석 결과를 보고하는 글이 아니라 사람을 소개하는 말로 쓴다.
"니쥬가 보기에 {topic} ~을 좋아하는 사람이에요"처럼 이 사람이 무엇을 좋아하고
어떤 데 관심이 있고 어떤 시간을 즐기는 사람인지를 주어로 세워서 말한다.
"대화에서", "단서", "분석", "취향의 결", "파악했어요" 같은 분석 과정을 드러내는 말은 쓰지 않는다.
입력의 항목 이름을 나열하지 말고, 그 항목들이 함께 가리키는 취향을 묶어서 쓴다.
예를 들어 "캠핑", "핸드드립", "조용한 카페"가 있으면 "캠핑에 관심이 있어요"가 아니라
"자연에서 보내는 시간과 직접 내린 커피를 좋아하고, 조용한 곳에서 쉬는 걸 즐겨요"처럼 쓴다.
서로 다른 항목을 한 장면으로 엮어 실제로 그런 일이 있었던 것처럼 쓰지 않는다.
("빈티지한 카페에서 따뜻한 조명 아래 차를 마시는 모습"처럼 입력에 없는 장면을 만들지 않는다)
항목 사이의 공통점과 좋아하는 방식(혼자인지, 손으로 만드는지, 새로운 걸 찾는지 등)을 짚는다.
대화에서 나온 가게, 메뉴, 단계, 가사 같은 구체적인 이름과 표현은 그대로 옮기지 않는다.
그것들이 함께 보여 주는 더 넓은 취향으로 바꿔서 쓴다. 예를 들어 "엽떡", "약간 매운맛 떡볶이"가
있으면 "엽떡을 좋아해요"가 아니라 "적당히 매콤한 분식을 즐겨요"처럼 쓴다.
취향은 무엇을 좋아하는지, 관심사는 무엇에 관심을 두는지다. 둘이 섞이지 않게 각각 그려서 쓴다.
입력에 있는 취향과 관심사를 중심으로 소개하고, 아래 마지막 문장 말고는 짐작해서 덧붙이지 않는다.

[좋아할 것 같은 것]
마지막 문장에서 입력 항목들이 함께 가리키는 쪽으로, 아직 말하지 않았지만 좋아할 것 같은 것 하나를
"~도 좋아할 것 같아요"로 덧붙인다. 서로 다른 항목 두 개 이상에서 근거가 보일 때만 쓰고,
이미 나온 관심사와 가까운 것으로 고른다. 근거가 약하면 이 문장은 쓰지 않는다.
무엇을 좋아할 것 같은지는 이 사람의 입력 항목에서만 찾고, 이 지시문에 나온 예시에서 가져오지 않는다.

[별로라고 한 것]
avoids는 이 사람이 별로라고 한 것이다. 이것만 따로 문장으로 쓰지 않는다.
"싫어해요", "별로예요", "피해요", "꺼려요"처럼 싫다는 뜻의 말은 쓰지 않는다.
좋아하는 것을 소개하는 문장 안에서 "북적이는 곳보다 조용한 데를 좋아해요"처럼 대비로만 드러내고,
함께 말할 좋아하는 항목이 없으면 쓰지 않는다.

[입력]
interests는 관심사를 분야별로, tastes는 취향을 축별로 묶은 것이다.
앞에 있는 묶음일수록 이야기가 많이 나온 것이다.
한 축에 서로 다른 관심사의 항목이 모여 있으면 그 사람의 뚜렷한 모습이다.
예를 들어 사회/인원 축에 "혼자 가는 캠핑"과 "혼자 보는 영화"가 함께 있으면
캠핑이나 영화 하나가 아니라 혼자 온전히 보내는 시간을 좋아하는 사람으로 소개한다.
항목을 빠짐없이 나열하려 하지 말고, 여러 묶음에 걸친 공통점을 중심으로 쓴다.
동기 묶음(휴식·재충전, 몰입 등)이 있으면 이 사람이 그 일들에서 무엇을 얻는지를 소개의 중심에 둔다.
target은 그 취향이 어느 관심사에 대한 것인지다. ("혼자 가는 캠핑"의 target은 캠핑)
field와 axis는 해석을 돕는 분류 이름이다. "사회/인원" 같은 분류 이름은 글에 쓰지 않는다.

[형식]
한국어 존댓말("~해요", "~이에요")로 3~5문장, 한 문단으로 쓴다. 목록, 마크다운, 이모지는 쓰지 않는다.
첫 문장은 "니쥬가 보기에 {topic}"으로 시작해서 이 사람을 한마디로 소개한다.
{address_rule}
단정하는 판정보다 니쥬가 본 인상으로 말한다("~하는 사람이에요", "~을 좋아해요").
담백하게 쓴다. "피로를 녹여내는", "스스로를 다독이는", "소중한 시간", "특별한 순간"처럼
감성적인 수식어나 거창한 의미 부여는 하지 않는다.
"손이 가요", "마음이 가요"처럼 돌려 말하는 관용 표현보다 "좋아해요", "즐겨요",
"관심이 많아요"처럼 바로 읽히는 말로 쓴다.
항목이 하나뿐이면 무리하게 넓히지 말고 2문장으로 짧게 쓴다.

[하지 않는 것]
입력에 없는 구체적인 사실(장소, 사람, 횟수, 경력)을 지어내지 않는다.
혼자인지 누구와인지, 언제 어디서인지는 입력 항목에 있을 때만 쓴다.
누구와 함께하는 항목이 있다고 해서 다른 활동을 "혼자", "조용히" 한다고 대비해 쓰지 않는다.
성격 유형이나 내향적·외향적 같은 성향 판정은 하지 않는다.
"꼼꼼한 면", "차분한 성격"처럼 성격을 평가하는 말도 쓰지 않는다.
건강·질병·복용약, 종교, 정치 성향, 성적 지향, 가족 관계와 갈등, 경제 사정, 알레르기,
구매하지 못한 이유, 소유물, 생활환경, 내부 점수는 언급하지 않는다.
입력 항목이 이런 민감한 내용에 해당하면 그 항목은 빼고 쓴다.
상품이나 브랜드를 추천하지 않는다.

[다 쓰고 나서 확인할 것]
아래에 걸리는 문장이 있으면 고친 뒤 최종 글만 쓴다.
- 입력에 없는 "혼자", "조용히", "차분히" 같은 상황이나 성향을 붙였는가
- avoids 항목을 "피해요", "싫어해요"처럼 직접 말했는가
- "소중한", "마음이 가는", "손이 가는" 같은 감성 표현이나 관용 표현을 썼는가
- 가게, 메뉴, 가사 같은 구체적인 이름을 그대로 옮겼는가
- 서로 다른 항목을 엮어 입력에 없는 장면을 만들었는가"""
_UNNAMED_ADDRESS_RULE = (
    '이 사람은 "사용자"라고 부른다. "이분", "고객", 이름 같은 다른 호칭은 쓰지 않는다.'
)
_NAMED_ADDRESS_RULE = (
    '이 사람은 "{name}님"으로 부른다. "사용자", "이분", "고객" 같은 다른 호칭은 쓰지 않는다.'
)


def friend_summary_system(name: str | None = None) -> str:
    """요약 시스템 프롬프트. name 은 username 원문(display_name)이고 성을 떼지 않는다.

    "님"은 받침이 있어 이름이 무엇이든 "님은"으로 쓴다. 이름이 없으면 "사용자는"이다.
    """
    if not name:
        return _FRIEND_SUMMARY_TEMPLATE.format(topic="사용자는", address_rule=_UNNAMED_ADDRESS_RULE)
    return _FRIEND_SUMMARY_TEMPLATE.format(
        topic=f"{name}님은", address_rule=_NAMED_ADDRESS_RULE.format(name=name)
    )


FRIEND_SUMMARY_SYSTEM = friend_summary_system()


def build_friend_summary_messages(
    profile: ProfileState,
    *,
    name: str | None = None,
) -> list[dict[str, str]]:
    # 친구 공개 가능한 칸의 저장 항목 전체를 강도 순으로 넘긴다. 노출 한도 밖의 취향도
    # 요약에는 들어가야 결이 보인다. context는 해석용이고 프롬프트가 인용을 막는다.
    interests: list[dict[str, object]] = []
    tastes: list[dict[str, object]] = []
    for path, signals in friend_signal_groups(
        profile, _SUMMARY_FIELD_LABELS, limit=SUMMARY_CLUE_LIMIT
    ):
        items = [
            {
                "kind": _SUMMARY_FIELD_LABELS[signal.field],
                "value": signal.value,
                # 발화 원문은 다른 사람에게 보일 글에 새어 나갈 수 있어 넘기지 않는다.
                **({"target": signal.target} if signal.target else {}),
            }
            for signal in signals
        ]
        label = "/".join(path[1:]) if path[0] in {TASTE_ROOT, INTEREST_ROOT} else "기타"
        if signals[0].field == TasteField.PREFERENCES:
            tastes.append({"axis": label, "items": items})
        else:
            interests.append({"field": label, "items": items})
    # 별로라고 한 것은 친구 공개 항목이 아니라서 글에 대비로만 드러낸다(프롬프트 [별로라고 한 것]).
    # 민감한 내용은 추출 단계에서 이미 저장하지 않는다.
    avoids = [
        {"value": signal.value}
        for signal in sorted(
            (signal for signal in profile.active_signals() if signal.field == TasteField.DISLIKES),
            key=taste_rank_score,
            reverse=True,
        )[:SUMMARY_AVOID_LIMIT]
    ]
    return [
        {"role": "system", "content": friend_summary_system(name)},
        {
            "role": "user",
            "content": json.dumps(
                {"interests": interests, "tastes": tastes, "avoids": avoids},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
