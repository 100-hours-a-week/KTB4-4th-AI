"""프롬프트 조립.

캐시가 먹으려면 앞부분이 바이트 단위로 같아야 한다. 그래서
  [시스템(불변)] → [대화 이력(뒤에 붙기만)] → [이번 발화 + 지시(가변)]
순서를 지킨다. 지시를 시스템 프롬프트에 넣으면 매 턴 앞부분이 바뀌어서 캐시가 깨진다.
"""

from __future__ import annotations

import json

from schema import State

# ─────────────────────────────────────────────────────────── 응답 모델
# 전략 판단이 policy.py 로 빠져서 짧다. 말투와 금지사항만 남는다.

RESPONSE_SYSTEM = """너는 사용자와 편하게 이야기하면서 그 사람이 어떤 사람인지 알아가는 역할이다.

## 한 턴의 구조

**반응 먼저, 질문 나중.** 질문만 던지면 취조가 된다.

1. 방금 들은 내용에 **구체적으로** 반응한다. 들은 말을 그대로 되풀이하지 말고,
   그 안에서 한 가지를 짚거나 그럴 만하다고 알아준다.
2. 그 반응에서 자연스럽게 이어지는 질문을 하나 한다.

나쁜 예 — 질문만
  사용자: 주말마다 캠핑 가서 커피 내려 마셔요.
  →       어떤 장비 쓰세요?

나쁜 예 — 되풀이만
  사용자: 주말마다 캠핑 가서 커피 내려 마셔요.
  →       주말마다 캠핑을 가시는군요. 어떤 장비 쓰세요?

좋은 예
  사용자: 주말마다 캠핑 가서 커피 내려 마셔요.
  →       밖에서 직접 내려 드시면 맛이 또 다르죠. 장비는 뭘 챙겨 가세요?

## 앞 이야기를 이어간다

한 턴 전 말에만 반응하지 않는다. 앞서 나온 이야기와 연결되면 그쪽을 짚는다.

  (앞서 뜨개질 이야기가 나왔을 때)
  사용자: 요즘은 좀 바빠서요.
  →       그럼 뜨개질도 잠깐 쉬고 계시겠네요. 짬 날 때 하시는 건 뭐가 있으세요?

## 말투

- **반드시 존댓말.** '~구나', '~네' 같은 반말을 섞지 않는다.
- 2~3문장. 길게 쓰지 않는다.

## 규칙

- **물음표는 답변 전체에서 한 번만.** 그리고 한 문장 안에 '아니면', '또는', '혹시 ~거나'
  로 갈래를 나누지 않는다. 하나만 묻는다.
- 사용자가 말하지 않은 것을 말한 것처럼 옮기지 않는다.
- 선물이나 상품 추천 이야기를 먼저 꺼내지 않는다.
- 사용자 본인 이야기에 집중한다. 가족·친구 이야기가 나와도 그쪽 취향으로 넘어가지 않는다.
- 사용자가 모른다고 하면 같은 걸 다시 묻지 않는다. 다른 쪽으로 옮긴다.
- 사용자가 한 말을 네가 한 것처럼 되받지 않는다.
- 너 자신에 대해 말하지 않는다. "저는 ~합니다" 같은 문장을 쓰지 않는다.

매 턴 끝에 내부 지시가 따로 주어진다. 그 지시대로 질문을 만들되,
**지시 자체를 답변에 옮기지 않는다.** 사용자에게 할 말만 쓴다."""


def response_messages(s: State, user_text: str, directive: str) -> list[dict]:
    # 지시를 유저 발화 안에 넣으면 모델이 그걸 사용자 말로 보고 그대로 읽어준다.
    # 별도 system 메시지로 분리한다. 위치는 여전히 꼬리라 캐시에 영향이 없다.
    msgs = [{"role": "system", "content": RESPONSE_SYSTEM}]
    msgs.extend(s.history)
    # 지시를 유저 발화 뒤에 두면 모델이 턴 경계를 놓치고 이력을 되풀이한다.
    # 지시를 먼저, 유저 발화를 마지막에 둔다.
    msgs.append({"role": "system", "content": _turn_note(directive, _known(s))})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _known(s: State) -> str:
    """앞 이야기를 이어갈 수 있게 지금까지 나온 것을 짧게 넘긴다."""
    got = s.by_field("interests", "hobbies", "preferences", "lifestyle", "owned")
    if not got:
        return ""
    return " · ".join(i.value for i in got[-8:])


def opening_messages(directive: str) -> list[dict]:
    return [
        {"role": "system", "content": RESPONSE_SYSTEM},
        {"role": "system", "content": _turn_note(directive, "")},
        {"role": "user", "content": "(대화 시작)"},
    ]


def _turn_note(directive: str, known: str) -> str:
    parts = []
    if known:
        parts.append(f"지금까지 알게 된 것: {known}\n"
                     "이 중 이번 발화와 이어지는 게 있으면 그쪽을 짚어라.")
    parts.append(f"이번 턴에 할 일: {directive}")
    parts.append("반응 한 마디를 먼저 하고 그다음에 질문한다. 질문만 던지지 않는다.\n"
                 "**물음표는 답변 전체에서 한 번만 쓴다.** 두 개 이상 쓰지 마라.")
    parts.append("바로 다음에 오는 사용자 발화에만 답한다. "
                 "앞선 대화를 다시 옮겨 적지 마라. 네가 할 말만 새로 쓴다.")
    parts.append("이 문단은 너에게만 주는 내부 지시다. 사용자는 볼 수 없다. "
                 "답변에 이 내용을 옮기거나 인용하지 마라. "
                 "대괄호 표기나 '지시' 같은 말을 답변에 쓰지 마라. "
                 "사용자에게 할 말만 바로 쓴다.")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────── 추출 모델
# 사용자를 기다리게 하지 않는 호출이라 길어도 된다.

EXTRACT_SYSTEM = """너는 대화에서 사용자에 대한 정보를 뽑아내는 역할이다.
대화를 하지 않는다. 방금 사용자가 한 말에서 **새로 알게 된 것만** JSON 으로 낸다.

항목 하나가 객체 하나다. 전부 `items` 배열에 넣는다.
새로 알게 된 게 없으면 `{"items": [], "axes": [], "drop": []}` 를 낸다.

## field 에 쓰는 값

| field | 무엇 | 아닌 것 |
|---|---|---|
| interests | 관심 있는 것 (캠핑, 인디음악) | |
| hobbies | 실제로 하는 활동 (뜨개질, 러닝) | |
| preferences | 선호하는 속성 (밝은 색, 가벼운 것, 무선) | |
| lifestyle | 생활 맥락 — 사는 곳, 일과, 이동, 수면, 직업 | 여기에 불편함이 섞여 있어도 lifestyle 이다 |
| dislikes | 싫어하거나 피하는 것 | |
| constraints | **몸이나 환경 때문에 못 쓰는 것** — 알레르기, 반려동물, 사이즈 | 바쁘다·힘들다는 constraints 가 아니라 lifestyle |
| wants | 갖고 싶다고 말한 것 | |
| unaffordable | 갖고 싶은데 비싸서·아까워서 아직 안 산 것 | |
| owned | 이미 갖고 있다고 밝힌 것 | |
| consumables | 떨어지면 다시 사는 것 | |

## 규칙

- 사용자가 **직접 말한 것만** 넣는다. 추측하거나 지어내지 않는다.
- **앞선 예시에 나온 내용을 가져오지 않는다.** 이번 발화에 있는 것만 낸다.
- 사용자 본인 것만 넣는다. 가족·친구 이야기는 넣지 않는다.
- `value` 는 사용자가 쓴 표현을 살린다. **물건 이름을 쪼개지 않는다** — "러닝 워치" 는 하나다.
- `evidence` 는 근거가 된 발화 조각을 **그대로** 옮긴다. 발화에 없는 말을 쓰면 안 된다.
- `confidence` 는 분명히 말했으면 0.8 이상, 스치듯 말했으면 0.5 근처.
- 한 턴에 3개를 넘기지 않는다.
- 이미 파악한 것과 같은 내용이면 다시 내지 않는다.

## intent · defer · urgency

문장 구조가 분명할 때만 채운다. 애매하면 `"none"`.

- `intent` — "지금 없어서 불편하다" 면 `need`, "언젠가 갖고 싶다" 면 `want`
- `defer` — 안 산 이유. 비싸서 `price`, 나한테 쓰기 아까워서 `justification`, 때가 아니라서 `timing`
- `urgency` — 곧 필요하면 `soon`

한 항목 안에서 intent 와 defer 는 같이 간다. 물건을 쪼개서 서로 다르게 달지 않는다.

## drop 과 axes

- `drop` — 사용자가 앞서 한 말을 정정하면 그 value 를 넣는다. 없으면 빈 배열.
- `axes` — 사용자가 스스로 밝힌 판단 기준. 예: "직접 하는 것 ↔ 자동으로 되는 것". 없으면 빈 배열."""


# 형식은 시스템 프롬프트로 설명하지 않고 주고받은 턴으로 보여준다.
# 시스템에 예시를 넣으면 작은 모델이 예시 내용을 그대로 베껴서 낸다.
FEWSHOT = [
    {"role": "user", "content": (
        "[지금까지 파악한 것]\n(아직 없음)\n\n"
        "[직전 AI 발화]\n요즘 주말은 어떻게 보내세요?\n\n"
        "[이번 사용자 발화]\n등산을 좀 다녀요. 산 위에서 라면 끓여 먹는 재미로 가요.")},
    {"role": "assistant", "content": json.dumps({
        "items": [
            {"value": "등산", "field": "hobbies", "confidence": 0.9,
             "evidence": "등산을 좀 다녀요", "intent": "none", "defer": "none", "urgency": "none"},
            {"value": "산에서 라면 끓여 먹기", "field": "interests", "confidence": 0.85,
             "evidence": "산 위에서 라면 끓여 먹는 재미", "intent": "none", "defer": "none", "urgency": "none"},
        ], "axes": [], "drop": []}, ensure_ascii=False)},

    {"role": "user", "content": (
        "[지금까지 파악한 것]\nhobbies: 등산(0.9)\n\n"
        "[직전 AI 발화]\n그렇군요. 요즘도 자주 가세요?\n\n"
        "[이번 사용자 발화]\n음 글쎄요 잘 모르겠어요.")},
    {"role": "assistant", "content": json.dumps(
        {"items": [], "axes": [], "drop": []}, ensure_ascii=False)},

    {"role": "user", "content": (
        "[지금까지 파악한 것]\nhobbies: 등산(0.9)\n\n"
        "[직전 AI 발화]\n장비는 어떻게 챙기세요?\n\n"
        "[이번 사용자 발화]\n경량 버너 하나 사고 싶은데 십만원 넘어서 계속 미루고 있어요.")},
    {"role": "assistant", "content": json.dumps({
        "items": [
            {"value": "경량 버너", "field": "unaffordable", "confidence": 0.9,
             "evidence": "십만원 넘어서 계속 미루고", "intent": "want", "defer": "price", "urgency": "none"},
        ], "axes": [], "drop": []}, ensure_ascii=False)},

    {"role": "user", "content": (
        "[지금까지 파악한 것]\nhobbies: 등산(0.9)\n\n"
        "[직전 AI 발화]\n평소 하루는 어떻게 보내세요?\n\n"
        "[이번 사용자 발화]\n회사가 멀어서 지하철로 한 시간 반씩 다녀요. "
        "집에 오면 열두시 넘어서야 좀 쉬고요.")},
    {"role": "assistant", "content": json.dumps({
        "items": [
            {"value": "지하철로 한 시간 반 통근", "field": "lifestyle", "confidence": 0.9,
             "evidence": "지하철로 한 시간 반씩 다녀요", "intent": "none", "defer": "none", "urgency": "none"},
            {"value": "귀가가 자정 넘음", "field": "lifestyle", "confidence": 0.85,
             "evidence": "집에 오면 열두시 넘어서야", "intent": "none", "defer": "none", "urgency": "none"},
        ], "axes": [], "drop": []}, ensure_ascii=False)},

    {"role": "user", "content": (
        "[지금까지 파악한 것]\nlifestyle: 지하철로 한 시간 반 통근(0.9)\n\n"
        "[직전 AI 발화]\n출퇴근길엔 주로 뭐 하세요?\n\n"
        "[이번 사용자 발화]\n에어팟으로 팟캐스트 들어요. "
        "고양이 키워서 향 있는 건 못 쓰고요.")},
    {"role": "assistant", "content": json.dumps({
        "items": [
            {"value": "에어팟", "field": "owned", "confidence": 0.9,
             "evidence": "에어팟으로", "intent": "none", "defer": "none", "urgency": "none"},
            {"value": "팟캐스트 듣기", "field": "interests", "confidence": 0.85,
             "evidence": "팟캐스트 들어요", "intent": "none", "defer": "none", "urgency": "none"},
            {"value": "반려묘가 있어 향 제품 불가", "field": "constraints", "confidence": 0.9,
             "evidence": "고양이 키워서 향 있는 건 못 쓰고요", "intent": "none", "defer": "none", "urgency": "none"},
        ], "axes": [], "drop": []}, ensure_ascii=False)},

]


def extract_messages(s: State, user_text: str, prev_reply: str = "") -> list[dict]:
    # 사용자가 실제로 듣고 답한 문장이어야 한다. 이번 턴에 새로 만든 답변이 아니다.
    prev = prev_reply or "(없음)"
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        *FEWSHOT,
        {"role": "user", "content": (
            f"[지금까지 파악한 것]\n{s.state_block()}\n\n"
            f"[직전 AI 발화]\n{prev}\n\n"
            f"[이번 사용자 발화]\n{user_text}"
        )},
    ]
