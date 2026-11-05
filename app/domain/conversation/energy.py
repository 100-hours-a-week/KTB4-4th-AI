"""companion 대화에서 사용자가 얼마나 신나서 이야기하는지(에너지)를 0~1로 추정한다.

신호는 답 길이(이 사용자의 평균 대비), 웃음과 감탄, 주도성(니쥬에게 되묻기), 회피 표현,
그리고 판단 모델이 매긴 답 깊이다. 길이, 표현, 주도성은 이번 발화에서 바로 계산해 이번 응답에 쓰고,
답 깊이는 추출과 같이 끝나므로 턴을 기록할 때 더해 다음 턴부터 반영된다.
"""

from __future__ import annotations

import re

from app.domain.conversation.models import ConversationMove, ConversationState

# 최근 턴에 주는 가중치. 나머지는 이전 에너지에서 온다.
ENERGY_RECENT_WEIGHT = 0.6
# 이 아래면 에너지가 낮은 턴으로 센다.
LOW_ENERGY = 0.35
# 선택지에 고르기만 한 답의 에너지 하한. 짧게 고르는 게 정상이라 지친 것으로 보지 않는다.
CHOICE_ANSWER_ENERGY = 0.5
# 평균 길이가 없을 때(첫 답) 이 길이면 길이 점수를 꽉 채운다.
FIRST_ANSWER_FULL_LENGTH = 30

_LAUGHTER = re.compile(r"ㅋㅋ|ㅎㅎ|ㄷㄷ|!|헐|대박|완전|진짜|너무|ㄹㅇ|최고")
# 긍정·부정의 짧은 답은 회피가 아니다. 질문에 충실하게 답했을 수 있다.
_ACKNOWLEDGEMENT = re.compile(r"(?:네|넹|넵|응|ㅇㅇ|아뇨|아니요|그렇죠|그쵸)[.~…!]*")
_SHORT_AVOIDANCE = re.compile(r"(?:그냥(?:요)?|몰라(?:요)?|딱히(?:요)?|글쎄(?:요)?|음+|흠+)[.~…!]*")
_UNSURE_RESPONSE = re.compile(
    r"(?:음[, ]*|저도\s*|저는\s*|아직\s*|잘\s*)*"
    r"(?:모르겠(?:어(?:요)?|는데(?:요)?|어요|네요)|몰라(?:요)?|글쎄(?:요)?|"
    r"딱히(?:요|\s*없(?:어(?:요)?|는데(?:요)?|네요))?|"
    r"(?:별\s*거|생각나는\s*(?:게|건))\s*없(?:어(?:요)?|네요|는데(?:요)?))"
    r"[.~…!]*"
)
_REFUSAL = re.compile(
    r"(?:(?:그건|이건|지금은?|별로|그냥)\s*)*(?:(?:그|이)\s*)?"
    r"(?:말|얘기|이야기|대답|답)(?:은|는|을|를)?\s*하(?:기|고)\s*"
    r"(?:싫(?:어(?:요)?|은데(?:요)?)|싶지\s*않(?:아(?:요)?|네요))[.~…!]*"
)
_ASKS_BACK = re.compile(r"\?|니쥬는|니쥬도|너는|넌 |넌$|님은|님도")
# 물음표가 있어도 니쥬에게 묻는 게 아니라 니쥬 말에 어리둥절하거나 불만인 표현.
_PUZZLED = re.compile(r"어쩌라|뭐라는|뭔\s*소리|무슨\s*소리|무슨\s*말(이|인)|그래서\s*뭐")
# 니쥬의 말이나 짐작을 사용자가 고치는 표현. "아니요" 한마디는 평범한 답일 수 있어서 넣지 않고,
# "장난 아니야", "~뿐만 아니라"처럼 고치는 뜻이 없는 말도 뺀다.
_CORRECTION = re.compile(
    r"^아니[ ,]|(?<!장난 )(?<!뿐만 )(?:아니야|아닌데|아니거든|아니라|아니고|아니에요|아닌\s*것)|"
    r"(?:했|그랬)잖아|했다니까"
)
# 문장 전체가 직접적인 종료 요청일 때만 인정한다. 활동을 그만두거나 미루는 말,
# 부정·인용·설명 속의 종료 표현은 대화 종료가 아니다.
_EXIT_PREFIX = r"(?:(?:그럼|이제|오늘은?|여기서|일단|그냥|저는|나는|우리|이만)\s*)*"
_EXIT_END = (
    r"(?:요|주세요|줘요|줘|해요|하자|할게요|할래요|하고\s*싶어요|"
    r"하고\s*싶어|할게|할래|하죠|하겠습니다)"
)
_EXIT_INTENT = re.compile(
    _EXIT_PREFIX
    + r"(?:(?:(?:이\s*)?(?:대화|얘기|이야기|채팅)(?:는|를|는요)?\s*)?"
    + r"(?:그만\s*"
    + _EXIT_END
    + r"|끝내(?:요|줘요?|주세요|자|고\s*싶어요|고\s*싶어|겠습니다))|"
    + r"(?:다음에|나중에)\s*(?:다시\s*)?(?:얘기|이야기|대화)(?:해요|하자|할게요|할래요)|"
    + r"나갈게(?:요)?|갈게(?:요)?|됐(?:어요|어))"
    + r"[.~…!]*"
)


def wants_to_exit(utterance: str) -> bool:
    return any(_EXIT_INTENT.fullmatch(part.strip()) for part in re.split(r"[.!?\n]+", utterance))


def unsure(utterance: str) -> bool:
    """답 전체가 불확실한 경우. 제목·이름을 잊은 구체적인 이야기는 제외한다."""
    # 질문을 되받은 뒤 답한 경우에도 답 부분을 본다("좋아하는 거요? 딱히 없는데").
    answer = utterance.rsplit("?", 1)[-1].strip() or utterance.strip()
    return bool(_UNSURE_RESPONSE.fullmatch(answer))


def avoids_answer(utterance: str) -> bool:
    """회피가 명확한 답. 짧은 장소·이름·긍정·부정 답변은 여기에 포함하지 않는다."""
    text = utterance.strip()
    return bool(_SHORT_AVOIDANCE.fullmatch(text) or declines_topic(text) or unsure(text))


def declines_topic(utterance: str) -> bool:
    """특정 이야기에 답하고 싶지 않다는 직접 표현. 대화 전체의 종료와는 구별한다."""
    return bool(_REFUSAL.fullmatch(utterance.strip()))


def asks_back(utterance: str) -> bool:
    """사용자가 니쥬에게 되물었는지. 되묻기는 대화가 재밌다는 강한 신호다."""
    return (
        bool(_ASKS_BACK.search(utterance))
        and not _PUZZLED.search(utterance)
        and not unsure(utterance)
    )


def corrects(utterance: str) -> bool:
    """사용자가 니쥬의 말을 고치거나 니쥬 말에 어리둥절해하는지."""
    return bool(_CORRECTION.search(utterance) or _PUZZLED.search(utterance))


def _length_score(state: ConversationState, utterance: str) -> float:
    length = len(utterance.strip())
    if state.user_length_avg is None or state.user_length_avg <= 0:
        return min(length / FIRST_ANSWER_FULL_LENGTH, 1.0)
    # 원래 짧게 쓰는 사람을 이탈로 오해하지 않도록 이 사람 평균에 대한 비율로 본다.
    ratio = length / state.user_length_avg
    return max(0.0, min((ratio - 0.25) / 0.75, 1.0))


def turn_energy(
    state: ConversationState,
    utterance: str,
    *,
    answer_depth: float | None = None,
) -> float:
    """이번 발화 하나의 에너지. state는 이번 발화를 기록하기 전 상태여야 한다."""
    text = utterance.strip()
    # 회피 없이 짧게 답한 것까지 낮게 보지 않도록 기본값을 둔다.
    score = 0.3 + 0.5 * _length_score(state, text)
    if _LAUGHTER.search(text):
        score += 0.15
    if asks_back(text):
        score += 0.2
    avoiding = avoids_answer(text)
    if avoiding:
        score -= 0.35
    score = max(0.0, min(score, 1.0))
    if answer_depth is not None:
        score = 0.7 * score + 0.3 * max(0.0, min(answer_depth / 2, 1.0))
    if not avoiding:
        # 답변 길이나 깊이 점수만으로 정확한 짧은 답을 이탈 신호로 만들지 않는다.
        score = max(score, LOW_ENERGY)
        if _answers_choice(state) or _ACKNOWLEDGEMENT.fullmatch(text):
            score = max(score, CHOICE_ANSWER_ENERGY)
    return score


def _answers_choice(state: ConversationState) -> bool:
    """이번 발화가 니쥬가 준 선택지에 대한 답인지."""
    return state.last_move == ConversationMove.CHOICE


def blended_energy(state: ConversationState, current: float) -> float:
    if state.energy is None:
        return current
    return ENERGY_RECENT_WEIGHT * current + (1 - ENERGY_RECENT_WEIGHT) * state.energy


def record_energy(
    state: ConversationState,
    utterance: str,
    *,
    answer_depth: float | None = None,
) -> float:
    """이번 발화로 에너지와 평균 길이를 갱신하고, 이번 턴의 에너지를 돌려준다."""
    current = turn_energy(state, utterance, answer_depth=answer_depth)
    state.energy = blended_energy(state, current)
    state.peak_energy = max(state.peak_energy, state.energy)
    state.low_energy_streak = state.low_energy_streak + 1 if current < LOW_ENERGY else 0
    # 고르기 답은 짧아서 평균에 넣으면 평소 답이 길어 보이게 된다.
    if _answers_choice(state):
        return current
    previous_turns = sum(1 for turn in state.history if turn.role == "user")
    length = len(utterance.strip())
    if state.user_length_avg is None:
        state.user_length_avg = float(length)
    else:
        state.user_length_avg += (length - state.user_length_avg) / (previous_turns + 1)
    return current
