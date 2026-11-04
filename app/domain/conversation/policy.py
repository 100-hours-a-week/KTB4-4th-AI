from __future__ import annotations

from app.domain.conversation.energy import (
    LOW_ENERGY,
    asks_back,
    corrects,
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
from app.domain.profile.memory import recheck_topic
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
# 한 화제에서 이어 묻는 꼬리 질문 수. 화제를 꺼낸 질문(첫인사, 화제 전환, 놀이, 되돌아가기)은
# 세지 않는다. 한 소재는 두 번까지만 묻고 다른 소재로 넘어간다. 다음 소재는 모델이 고른다.
TOPIC_FOLLOW_LIMIT = 2
# 화제를 넘길 때 최근 이만큼의 move 안에 선택지 move가 없으면 열린 질문 대신 선택지를 준다.
# 질문-답만 이어지는 흐름을 중간중간 바꾼다.
CHOICE_GAP = 6
# 지난 대화의 기억을 다시 확인하는 건 이 턴 이후 한 번만 한다. 첫인사에서 이미 지난 이야기를
# 꺼냈으니 대화 초반에는 이번 대화 이야기를 먼저 듣는다.
MIN_RECALL_TURN = 4

# 답이 이 길이 이하면 단답으로 본다.
# "치킨이요", "한참 있다가요"는 들고 "난 매운 거 좋아해"는 들지 않는다.
TERSE_ANSWER_LENGTH = 8


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
    # 앞 턴에서 예약한 마무리는 이번 발화의 에너지로 다시 판단하지 않는다.
    if state.pending_wrap is not None:
        return GoalDecision(ConversationGoal.WRAP, state.pending_wrap)
    # 이번 발화가 마지막 턴이면 답변을 만들기 전에 마무리로 정한다.
    turns_after = state.turn_count + 1 if utterance else state.turn_count
    if turns_after >= MAX_TURNS:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.MAX_CYCLES)
    if utterance and wants_to_exit(utterance):
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)

    current: float | None = None
    if utterance:
        current = turn_energy(state, utterance)
        low_streak = state.low_energy_streak + 1 if current < LOW_ENERGY else 0
    else:
        low_streak = state.low_energy_streak
    if low_streak >= COMPANION_WRAP_STREAK:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)

    # 진행률이 100%가 되면 끝낸다. 화면의 진행률과 실제 종료가 어긋나지 않게 한다.
    if recommendation_readiness(state).sufficient:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.SUFFICIENT)

    def chat(move: ConversationMove, scene: str | None = None) -> GoalDecision:
        return GoalDecision(ConversationGoal.CHAT, move=move, scene=scene)

    # 사용자가 니쥬의 말을 고치고 있으면 화제를 넘기지 않고 그 말부터 듣는다.
    if utterance and corrects(utterance) and not unsure(utterance):
        return chat(ConversationMove.FOLLOW)
    topic_done = topic_follow_count(state) >= TOPIC_FOLLOW_LIMIT
    # 사용자가 되물었으면 먼저 답하고 그 이야기를 따라간다.
    # 이 화제의 꼬리 질문을 다 썼으면 답한 뒤 다른 이야기로 넘어간다.
    if utterance and asks_back(utterance):
        if topic_done:
            return chat(ConversationMove.SHIFT)
        return chat(ConversationMove.FOLLOW)
    last_move = state.last_move
    # 답하기 어려워하거나 단답이 이어지면 다른 소재로 넘어가 선택지를 준다. 소재는 모델이 고른다.
    if utterance and last_move != ConversationMove.CHOICE:
        if unsure(utterance) or _keeps_answering_short(state, utterance):
            return chat(ConversationMove.CHOICE)
    if low_streak >= COMPANION_LIGHTEN_STREAK:
        if last_move != ConversationMove.CHOICE:
            return chat(ConversationMove.CHOICE)
        return chat(ConversationMove.ADD)
    if topic_done:
        recall = _recall_topic(state)
        if recall is not None:
            return chat(ConversationMove.RECALL, recall)
        if ConversationMove.CHOICE.value not in state.move_history[-CHOICE_GAP:]:
            return chat(ConversationMove.CHOICE)
        return chat(ConversationMove.SHIFT)
    return chat(ConversationMove.FOLLOW)


def topic_follow_count(state: ConversationState) -> int:
    """지금 화제에서 이어 물은 꼬리 질문 수. 질문 없이 보탠 턴(ADD)은 세지도 끊지도 않는다."""
    count = 0
    for move in reversed(state.move_history):
        if move == ConversationMove.FOLLOW:
            count += 1
        elif move != ConversationMove.ADD:
            break
    return count


def _keeps_answering_short(state: ConversationState, utterance: str) -> bool:
    """단답이 이어지는지. 선택지를 고른 답은 원래 짧아서 이번 답에서 세지 않는다."""
    if state.last_move == ConversationMove.CHOICE or len(utterance.strip()) > TERSE_ANSWER_LENGTH:
        return False
    previous = _recent_user_messages(state)[-1:]
    return bool(previous) and len(previous[0]) <= TERSE_ANSWER_LENGTH


def _recall_topic(state: ConversationState) -> str | None:
    """다시 확인할 지난 대화의 관심사. 이번 대화에서 이미 나온 것과 꺼낸 것은 빼고 한 번만 한다."""
    if state.memory is None or state.memory_rechecked or state.turn_count < MIN_RECALL_TURN:
        return None
    heard = [signal.value for signal in state.profile.signals]
    return recheck_topic(
        state.memory,
        state.last_active_at,
        exclude=[*heard, *state.memory_recalled],
    )


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
