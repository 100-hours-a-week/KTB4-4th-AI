from __future__ import annotations

from app.domain.conversation.energy import (
    LOW_ENERGY,
    asks_back,
    blended_energy,
    record_energy,
    turn_energy,
    wants_to_exit,
)
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationMove,
    ConversationState,
    GoalDecision,
    ReadinessResult,
    SessionStatus,
)
from app.domain.profile.merger import normalize_text
from app.domain.profile.models import LinkRole, SignalStatus, TasteField

MAX_TURNS = 20
# 충분하다고 보는 취향 개수. merger의 MAX_ACTIVE_QUERY_SIGNALS와 같아서 이 이상은 모으지 않는다.
MIN_QUERY_SIGNALS = 5
MIN_HIGH_CONFIDENCE_QUERY_SIGNALS = 2
HIGH_CONFIDENCE_THRESHOLD = 0.6
# 서로 다른 축에서 이만큼 취향이 나와야 추천에 쓸 만하다고 본다.
MIN_TASTE_AXES = 2
QUERY_PROGRESS_WEIGHT = 25
HIGH_CONFIDENCE_PROGRESS_WEIGHT = 25
TASTE_PROGRESS_WEIGHT = 20
GEAR_PROGRESS_WEIGHT = 15
_TOTAL_PROGRESS_WEIGHT = (
    QUERY_PROGRESS_WEIGHT
    + HIGH_CONFIDENCE_PROGRESS_WEIGHT
    + TASTE_PROGRESS_WEIGHT
    + GEAR_PROGRESS_WEIGHT
)

# move 선택 기준.
# 에너지가 낮은 턴이 이만큼 이어지면 놀이나 되돌아가기로 가볍게 바꾸고,
# 바꿔도 계속 낮으면 그만하고 싶은 것으로 본다.
COMPANION_LIGHTEN_STREAK = 2
COMPANION_WRAP_STREAK = 4
# 준비가 된 뒤 니쥬가 먼저 끝내는 때: 이 턴을 넘겼거나, 최고 에너지에서 이만큼 떨어졌을 때.
COMPANION_CLOSE_AFTER_TURN = 14
COMPANION_CLOSE_ENERGY_DROP = 0.2
MOVE_HISTORY_LIMIT = 8

# 에너지가 낮을 때 가볍게 던지는 소재. 한마디로 답할 수 있는 것만 두고,
# 부족한 추천 항목으로 화제를 고르지 않는다.
COMPANION_SCENES: tuple[str, ...] = (
    "이번 주에 제일 웃겼던 일",
    "요즘 자주 먹는 것",
    "요즘 자주 듣는 노래",
    "이번 주말에 하고 싶은 것",
    "요즘 빠져 있는 것",
    "최근에 처음 해 본 것",
    "요즘 자주 가는 곳",
    "쉬는 날 아침에 제일 먼저 하는 것",
    "요즘 폰으로 제일 많이 보는 것",
    "언젠가 한 번 해 보고 싶은 것",
)


def _query_signals(state: ConversationState):
    return [
        signal for signal in state.profile.active_signals() if signal.link_role == LinkRole.QUERY
    ]


def _taste_keys(state: ConversationState) -> set[tuple[str, ...]]:
    """지금까지 나온 취향의 축. 활성 한도 밖으로 밀린 취향도 센다. 축이 없으면 값 자체로 센다."""
    return {
        signal.taxonomy_path or (signal.normalized_value,)
        for signal in state.profile.signals
        if signal.field == TasteField.PREFERENCES and signal.status != SignalStatus.SUPERSEDED
    }


def taste_axis_count(state: ConversationState) -> int:
    return len(_taste_keys(state))


def _gear_known(state: ConversationState) -> bool:
    """좋아하는 방식이나 가진 물건이 하나라도 나왔는지."""
    return any(
        signal.field in {TasteField.PREFERENCES, TasteField.OWNED}
        for signal in state.profile.active_signals()
    )


def recommendation_readiness(state: ConversationState) -> ReadinessResult:
    """추천에 쓸 만큼 모였는지. 싫어하는 것은 필수로 두면 별로였던 경험을 캐묻게 돼서 뺀다."""
    query_signals = _query_signals(state)
    high_confidence_count = sum(
        signal.confidence >= HIGH_CONFIDENCE_THRESHOLD for signal in query_signals
    )
    missing: list[str] = []
    if len(query_signals) < MIN_QUERY_SIGNALS:
        missing.append("query_signals")
    if high_confidence_count < MIN_HIGH_CONFIDENCE_QUERY_SIGNALS:
        missing.append("high_confidence_query_signals")
    if taste_axis_count(state) < MIN_TASTE_AXES:
        missing.append("taste_axes")
    if not _gear_known(state):
        missing.append("gear")
    return ReadinessResult(sufficient=not missing, missing_signals=tuple(missing))


def conversation_progress(state: ConversationState) -> int:
    """준비 조건을 얼마나 채웠는지(0~100). 한 번 오른 진행률은 내리지 않는다."""
    if state.status in {
        SessionStatus.INPUT_LOCKED,
        SessionStatus.REVIEW,
        SessionStatus.CLOSED,
    }:
        return 100

    query_signals = _query_signals(state)
    high_confidence_count = sum(
        signal.confidence >= HIGH_CONFIDENCE_THRESHOLD for signal in query_signals
    )
    progress = (
        min(len(query_signals) / MIN_QUERY_SIGNALS, 1.0) * QUERY_PROGRESS_WEIGHT
        + min(high_confidence_count / MIN_HIGH_CONFIDENCE_QUERY_SIGNALS, 1.0)
        * HIGH_CONFIDENCE_PROGRESS_WEIGHT
        + min(taste_axis_count(state) / MIN_TASTE_AXES, 1.0) * TASTE_PROGRESS_WEIGHT
    )
    if _gear_known(state):
        progress += GEAR_PROGRESS_WEIGHT
    return max(round(progress * 100 / _TOTAL_PROGRESS_WEIGHT), state.progress_floor)


def _recent_user_messages(state: ConversationState) -> list[str]:
    return [turn.content.strip() for turn in state.history if turn.role == "user"][-3:]


def decide_goal(state: ConversationState, utterance: str) -> GoalDecision:
    """사용자 에너지와 최근 흐름을 보고 이번 턴의 반응 방식(move)을 고른다.

    위에서부터 차례로 보고 처음 맞는 move를 고른다. 추천 준비 상태는 종료 판단에만 쓰고,
    추출된 정보의 개수나 부족한 항목으로 질문·화제 전환을 정하지 않는다.
    utterance가 비어 있으면(턴을 기록한 뒤 다음 차례를 볼 때) 기록된 에너지로 판단한다.
    """
    if state.status != SessionStatus.ACTIVE:
        return GoalDecision(
            ConversationGoal.WRAP,
            state.completion_reason or CompletionReason.USER_EXIT,
        )
    if state.turn_count >= MAX_TURNS:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.MAX_CYCLES)
    if utterance and wants_to_exit(utterance):
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)

    if utterance:
        current = turn_energy(state, utterance)
        energy: float | None = blended_energy(state, current)
        low_streak = state.low_energy_streak + 1 if current < LOW_ENERGY else 0
    else:
        energy = state.energy
        low_streak = state.low_energy_streak
    if low_streak >= COMPANION_WRAP_STREAK:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)

    readiness = recommendation_readiness(state)
    if readiness.sufficient and _good_moment_to_close(state, energy):
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.SUFFICIENT)

    def chat(move: ConversationMove, scene: str | None = None) -> GoalDecision:
        return GoalDecision(ConversationGoal.CHAT, move=move, scene=scene)

    # 사용자가 되물었으면 먼저 답하고 그 이야기를 따라간다.
    if utterance and asks_back(utterance):
        return chat(ConversationMove.FOLLOW)
    last_move = state.last_move
    if low_streak >= COMPANION_LIGHTEN_STREAK:
        topic = _return_topic(state, utterance)
        if topic is not None and last_move != ConversationMove.RETURN:
            return chat(ConversationMove.RETURN, topic)
        if last_move != ConversationMove.PLAY:
            return chat(ConversationMove.PLAY, _pick_scene(state))
        return chat(ConversationMove.ADD)
    return chat(ConversationMove.FOLLOW)


def _good_moment_to_close(state: ConversationState, energy: float | None) -> bool:
    """준비가 됐어도 즐겁게 이어지면 끝내지 않는다.

    오래 했거나 에너지가 꺾이기 시작할 때 끝낸다.
    """
    if state.turn_count >= COMPANION_CLOSE_AFTER_TURN:
        return True
    return energy is not None and state.peak_energy - energy >= COMPANION_CLOSE_ENERGY_DROP


def _return_topic(state: ConversationState, utterance: str) -> str | None:
    """앞에서 가장 신나게 이야기한 관심사.

    최근에 다시 나왔거나 이미 돌아간 적 있으면 쓰지 않는다.
    """
    topic = state.best_topic
    if topic is None or topic in state.returned_topics:
        return None
    recent = [utterance, *_recent_user_messages(state)[-2:]]
    normalized = normalize_text(topic)
    if any(normalized and normalized in normalize_text(message) for message in recent):
        return None
    return topic


def _pick_scene(state: ConversationState) -> str | None:
    """사용하지 않은 참고 소재를 고른다. 실제 화제는 사용자 발화의 맥락이 결정한다."""
    offset = state.conversation_room_id % len(COMPANION_SCENES)
    rotated = COMPANION_SCENES[offset:] + COMPANION_SCENES[:offset]
    return next((scene for scene in rotated if scene not in state.used_scenes), None)


def record_companion_turn(
    state: ConversationState,
    *,
    utterance: str,
    move: ConversationMove | None,
    scene: str | None,
    answer_depth: float | None,
    new_interests: list[str],
) -> None:
    """턴이 끝난 뒤 에너지와 move 흐름을 기록한다.

    history에 이번 발화를 넣기 전에 불러야 평균 길이가 이번 발화 없이 비교된다.
    """
    current = record_energy(state, utterance, answer_depth=answer_depth)
    if new_interests and current >= state.best_topic_energy:
        state.best_topic = new_interests[0]
        state.best_topic_energy = current
    if move is None:
        return
    state.move_history = [*state.move_history, move.value][-MOVE_HISTORY_LIMIT:]
    if move == ConversationMove.RETURN and scene:
        state.returned_topics.append(scene)
    if move == ConversationMove.PLAY and scene:
        state.used_scenes.append(scene)


def known_query_values(state: ConversationState) -> set[str]:
    return {
        normalize_text(signal.value)
        for signal in state.profile.signals
        if signal.link_role == LinkRole.QUERY
    }
