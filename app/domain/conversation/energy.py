"""companion 대화에서 사용자가 얼마나 신나서 이야기하는지(에너지)를 0~1로 추정한다.

신호는 답 길이(이 사용자의 평균 대비), 웃음과 감탄, 주도성(니쥬에게 되묻기), 회피 표현,
그리고 판단 모델이 매긴 답 깊이다. 길이, 표현, 주도성은 이번 발화에서 바로 계산해 이번 응답에 쓰고,
답 깊이는 추출과 같이 끝나므로 턴을 기록할 때 더해 다음 턴부터 반영된다.
"""

from __future__ import annotations

import re

from app.domain.conversation.models import ConversationState

# 최근 턴에 주는 가중치. 나머지는 이전 에너지에서 온다.
ENERGY_RECENT_WEIGHT = 0.6
# 이 아래면 에너지가 낮은 턴으로 센다.
LOW_ENERGY = 0.35
# 평균 길이가 없을 때(첫 답) 이 길이면 길이 점수를 꽉 채운다.
FIRST_ANSWER_FULL_LENGTH = 30

_LAUGHTER = re.compile(r"ㅋㅋ|ㅎㅎ|ㄷㄷ|!|헐|대박|완전|진짜|너무|ㄹㅇ|최고")
# 답 전체가 이 말 하나뿐이면 회피로 본다. "네 맞아요 캠핑 좋아해요"처럼 뒤에 이야기가 붙으면 아니다.
_SHORT_AVOIDANCE = re.compile(
    r"^(그냥(요)?|몰라(요)?|딱히(요)?|글쎄(요)?|별로(요)?|없어(요)?|음+|흠+|ㅇㅇ|응|네|넹|넵|"
    r"아뇨|아니요|그렇죠|그쵸)[.~…!]*$"
)
_AVOIDANCE = re.compile(r"딱히|글쎄|모르겠|기억이? ?안 나|생각이? ?안 나|별거 없|별 거 없")
_ASKS_BACK = re.compile(r"\?|니쥬는|니쥬도|너는|넌 |넌$|님은|님도")
# 대화를 그만하고 싶다는 뜻이 분명한 표현.
_EXIT_INTENT = re.compile(
    r"그만\s*(할|하|해)|이제\s*됐|끝낼|끝내(자|요|줘)|나갈게|다음에\s*(할|얘기|이야기)|"
    r"나중에\s*(할|얘기|이야기)|이만\s*(할|갈)"
)


def wants_to_exit(utterance: str) -> bool:
    return bool(_EXIT_INTENT.search(utterance))


def asks_back(utterance: str) -> bool:
    """사용자가 니쥬에게 되물었는지. 되묻기는 대화가 재밌다는 강한 신호다."""
    return bool(_ASKS_BACK.search(utterance))


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
    if _SHORT_AVOIDANCE.match(text) or _AVOIDANCE.search(text):
        score -= 0.35
    score = max(0.0, min(score, 1.0))
    if answer_depth is not None:
        score = 0.7 * score + 0.3 * max(0.0, min(answer_depth / 2, 1.0))
    return score


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
    previous_turns = sum(1 for turn in state.history if turn.role == "user")
    length = len(utterance.strip())
    if state.user_length_avg is None:
        state.user_length_avg = float(length)
    else:
        state.user_length_avg += (length - state.user_length_avg) / (previous_turns + 1)
    return current
