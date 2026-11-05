from __future__ import annotations

from app.domain.conversation.context import ConversationDirection
from app.domain.conversation.energy import (
    LOW_ENERGY,
    asks_back,
    avoids_answer,
    corrects,
    declines_topic,
    record_energy,
    turn_energy,
    unsure,
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
from app.domain.profile.merger import normalize_text, taste_rank_score
from app.domain.profile.models import LinkRole, SignalStatus, TasteField

MAX_TURNS = 20
# 충분하다고 보는 취향 개수. merger의 MAX_ACTIVE_QUERY_SIGNALS와 같아서 이 이상은 모으지 않는다.
MIN_QUERY_SIGNALS = 6
# 확실한 취향은 여러 번 말했거나 반복하는 활동이면 그만큼 더 쳐 준다(taste_rank_score).
MIN_HIGH_CONFIDENCE_QUERY_SIGNALS = 3
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
MOVE_HISTORY_LIMIT = 8
# 이 횟수부터 다른 이야기로 넓힐지 검토한다. 실제 화제 경계는 move 횟수로 알 수 없으므로
# 강제 상한으로 쓰지 않는다. 새 이야기인지와 전환의 자연스러움은 응답 모델이 문맥으로 판단한다.
TOPIC_FOLLOW_LIMIT = 2


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


def _high_confidence_count(query_signals) -> int:
    return sum(taste_rank_score(signal) >= HIGH_CONFIDENCE_THRESHOLD for signal in query_signals)


def recommendation_readiness(state: ConversationState) -> ReadinessResult:
    """추천에 쓸 만큼 모였는지. 싫어하는 것은 필수로 두면 별로였던 경험을 캐묻게 돼서 뺀다."""
    query_signals = _query_signals(state)
    high_confidence_count = _high_confidence_count(query_signals)
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
    high_confidence_count = _high_confidence_count(query_signals)
    progress = (
        min(len(query_signals) / MIN_QUERY_SIGNALS, 1.0) * QUERY_PROGRESS_WEIGHT
        + min(high_confidence_count / MIN_HIGH_CONFIDENCE_QUERY_SIGNALS, 1.0)
        * HIGH_CONFIDENCE_PROGRESS_WEIGHT
        + min(taste_axis_count(state) / MIN_TASTE_AXES, 1.0) * TASTE_PROGRESS_WEIGHT
    )
    if _gear_known(state):
        progress += GEAR_PROGRESS_WEIGHT
    return min(max(round(progress * 100 / _TOTAL_PROGRESS_WEIGHT), state.progress_floor), 95)


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
    # 이번 발화가 마지막 턴이면 답변을 만들기 전에 마무리로 정한다.
    turns_after = state.turn_count + 1 if utterance else state.turn_count
    if turns_after >= MAX_TURNS:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.MAX_CYCLES)
    if utterance and wants_to_exit(utterance):
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)

    def chat(move: ConversationMove, scene: str | None = None) -> GoalDecision:
        return GoalDecision(ConversationGoal.CHAT, move=move, scene=scene)

    # 정정과 직접 질문부터 처리한다. 세부를 기억하지 못한다는 말이 섞여도 정정은 유효하다.
    # 최대 턴·명시적 종료는 위에서 처리하되, 준비 완료에 따른 종료는 한 턴 미룰 수 있다.
    if utterance and (corrects(utterance) or asks_back(utterance)):
        return chat(ConversationMove.FOLLOW)
    if state.pending_wrap is not None:
        return GoalDecision(ConversationGoal.WRAP, state.pending_wrap)

    current: float | None = None
    if utterance:
        current = turn_energy(state, utterance)
        low_streak = state.low_energy_streak + 1 if current < LOW_ENERGY else 0
    else:
        low_streak = state.low_energy_streak
    if low_streak >= COMPANION_WRAP_STREAK:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)

    # 추천 준비 완료와 대화 종료는 별개다. 현재 발화를 듣기 전에 항목 개수로 닫지 않는다.
    context = state.conversation_context
    if context and context.source_turn != state.turn_count:
        context = None
    developing = bool(context and context.direction == ConversationDirection.DEVELOPING)
    if not utterance and not developing and recommendation_readiness(state).sufficient:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.SUFFICIENT)

    last_move = state.last_move
    # 거절·반복 회피에는 같은 내용을 캐묻지 않고 앞말과 이어지는 쉬운 질문으로 바꾼다.
    if utterance and (
        declines_topic(utterance)
        or (avoids_answer(utterance) and last_move == ConversationMove.CHOICE)
    ):
        return chat(ConversationMove.SHIFT)
    if utterance and unsure(utterance):
        return chat(ConversationMove.CHOICE)
    if low_streak >= COMPANION_LIGHTEN_STREAK:
        if last_move != ConversationMove.CHOICE:
            return chat(ConversationMove.CHOICE)
        return chat(ConversationMove.SHIFT)
    # 직전 발화에서 기준·이유를 설명하는 중이라는 관찰은 참고하되, 최신 말이 우선이다.
    if developing and not avoids_answer(utterance):
        return chat(ConversationMove.FOLLOW)
    if context and context.direction == ConversationDirection.BROADEN:
        return chat(ConversationMove.SHIFT)
    if topic_follow_count(state) >= TOPIC_FOLLOW_LIMIT:
        return chat(ConversationMove.SHIFT)
    return chat(ConversationMove.FOLLOW)


def topic_follow_count(state: ConversationState) -> int:
    """최근 전환 검토 이후 FOLLOW 수. 실제 화제 수가 아니며 ADD는 세지도 끊지도 않는다."""
    count = 0
    for move in reversed(state.move_history):
        if move == ConversationMove.FOLLOW:
            count += 1
        elif move != ConversationMove.ADD:
            break
    return count


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
    if move in {ConversationMove.CHOICE, ConversationMove.SHIFT} and scene:
        state.used_scenes.append(scene)
    if move == ConversationMove.RECALL and scene:
        state.memory_recalled.append(scene)
        state.memory_rechecked = True


def known_query_values(state: ConversationState) -> set[str]:
    return {
        normalize_text(signal.value)
        for signal in state.profile.signals
        if signal.link_role == LinkRole.QUERY
    }
