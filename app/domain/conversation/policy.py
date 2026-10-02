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
    ConversationStyle,
    CoverageStatus,
    GoalArea,
    GoalDecision,
    ReadinessResult,
    SessionStatus,
    ThreadStage,
)
from app.domain.profile.merger import friend_signal_groups, normalize_text
from app.domain.profile.models import (
    AssessmentStatus,
    LinkRole,
    PreferenceAspect,
    ProfileSignal,
    SignalStatus,
    TasteField,
)
from app.domain.profile.taxonomy import (
    ASK_PRIORITY,
    TASTE_AXES_BY_KEY,
    TasteAxis,
    taste_axis_for_path,
)

MAX_TURNS = 20
# 충분하다고 보는 취향 개수. merger의 MAX_ACTIVE_QUERY_SIGNALS와 같아서 이 이상은 모으지 않는다.
MIN_QUERY_SIGNALS = 5
MIN_HIGH_CONFIDENCE_QUERY_SIGNALS = 2
DISENGAGED_MESSAGE_LENGTH = 10
HIGH_CONFIDENCE_THRESHOLD = 0.6
MAX_GOAL_ATTEMPTS = 3
# 기본 순서 이후 모자란 소지품·싫은 것을 다시 물을 수 있는 총 시도 횟수.
MAX_FOLLOW_UP_ATTEMPTS = 5
# 서로 다른 축에서 이만큼 취향이 나와야 추천에 쓸 만하다고 본다.
MIN_TASTE_AXES = 2
QUERY_PROGRESS_WEIGHT = 25
HIGH_CONFIDENCE_PROGRESS_WEIGHT = 25
TASTE_PROGRESS_WEIGHT = 20
GEAR_PROGRESS_WEIGHT = 15
EXCLUSION_PROGRESS_WEIGHT = 15
# 새 관심사 없이 같은 이야기로 이만큼 주고받으면 옆 화제로 넓힌다.
TOPIC_BROADEN_AFTER_TURNS = 2

# companion 대화의 move 선택 기준.
# 질문으로 끝난 응답이 이만큼 이어지면 다음 턴은 질문 없이 보태거나 되비춘다.
COMPANION_MAX_QUESTION_STREAK = 2
# 에너지가 낮은 턴이 이만큼 이어지면 놀이나 되돌아가기로 가볍게 바꾸고,
# 바꿔도 계속 낮으면 그만하고 싶은 것으로 본다.
COMPANION_LIGHTEN_STREAK = 2
COMPANION_WRAP_STREAK = 4
# 같은 화제에 이만큼 머물렀고 에너지가 떨어지고 있으면 건너간다. 에너지가 높으면 계속 머문다.
COMPANION_STAY_TURNS = 3
# 최고치보다 이만큼 떨어졌거나 이 값 아래면 에너지가 떨어지는 중으로 본다.
ENERGY_FALLING_DROP = 0.15
ENERGY_FALLING_LEVEL = 0.5
# 되비추기는 에너지가 이 이상일 때, 대화 전체에서 이 횟수까지만 한다.
REFLECT_BACK_MIN_ENERGY = 0.45
MAX_REFLECT_BACK = 2
REFLECT_BACK_FROM_TURN = 3
# 준비가 된 뒤 니쥬가 먼저 끝내는 때: 이 턴을 넘겼거나, 최고 에너지에서 이만큼 떨어졌을 때.
COMPANION_CLOSE_AFTER_TURN = 14
COMPANION_CLOSE_ENERGY_DROP = 0.2
MOVE_HISTORY_LIMIT = 8

# 건너가기와 놀이에서 쓰는 일상 장면. 장면마다 채워 줄 수 있는 추천 공백을 붙인다.
COMPANION_SCENES: tuple[tuple[str, frozenset[str]], ...] = (
    ("퇴근하고 집에 와서 제일 먼저 하는 것", frozenset({"query_signals"})),
    ("이번 주말에 하고 싶은 것", frozenset({"query_signals"})),
    ("요즘 꽂혀 있는 것", frozenset({"query_signals", "high_confidence_query_signals"})),
    ("나한테 주는 작은 보상", frozenset({"query_signals", "gear"})),
    ("요즘 장바구니에 담아 둔 것", frozenset({"query_signals", "high_confidence_query_signals"})),
    ("최근에 받고 기분 좋았던 것", frozenset({"gear", "query_signals"})),
    ("요즘 제일 손이 자주 가는 물건", frozenset({"gear"})),
    ("이 계절에 꼭 하고 싶은 것", frozenset({"query_signals"})),
    ("휴가가 생기면 하고 싶은 것", frozenset({"query_signals"})),
    ("요즘 은근히 귀찮거나 번거로운 일", frozenset({"exclusion"})),
)

# reflective 대화: 한 이야기 줄기에서 주고받는 최대 턴 수와 "왜" 질문 최대 횟수.
MAX_THREAD_TURNS = 4
MAX_WHY_QUESTIONS = 2
# 답의 깊이(0~2)가 이보다 낮으면 짧게 넘긴 답으로 보고 가벼운 질문으로 돌아온다.
SHALLOW_ANSWER_DEPTH = 0.6
# 짧게 넘기는 답이 이만큼 이어지면 대화를 그만하고 싶은 것으로 본다.
SHALLOW_STREAK_TO_WRAP = 3
# 되비추기에 쓰는 단서 묶음 수와 단서 수.
REFLECTION_GROUPS = 2
REFLECTION_CLUES = 4
_REFLECTION_FIELDS = (TasteField.INTERESTS, TasteField.HOBBIES, TasteField.PREFERENCES)
_STAGE_GOAL = {
    ThreadStage.WHAT: ConversationGoal.INTEREST,
    ThreadStage.HOW: ConversationGoal.TASTE,
    ThreadStage.WHY: ConversationGoal.DEEPEN,
    ThreadStage.CONTRAST: ConversationGoal.DISLIKE,
    ThreadStage.BRIDGE: ConversationGoal.BRIDGE,
}
_STAGE_ORDER = (ThreadStage.HOW, ThreadStage.WHY, ThreadStage.CONTRAST, ThreadStage.BRIDGE)

_GOAL_AREA = {
    ConversationGoal.INTEREST: GoalArea.INTEREST,
    ConversationGoal.INTEREST_VIA_ROUTINE: GoalArea.INTEREST,
    ConversationGoal.TASTE: GoalArea.TASTE,
    ConversationGoal.BRIDGE: GoalArea.INTEREST,
    ConversationGoal.DISLIKE: GoalArea.EXCLUSION,
    ConversationGoal.GEAR: GoalArea.GEAR,
    ConversationGoal.DEEPEN: GoalArea.DEEPEN,
}

_GOAL_FIELDS = {
    ConversationGoal.INTEREST: {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    },
    ConversationGoal.INTEREST_VIA_ROUTINE: {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    },
    ConversationGoal.TASTE: {TasteField.PREFERENCES},
    ConversationGoal.BRIDGE: {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    },
    ConversationGoal.DISLIKE: {TasteField.DISLIKES, TasteField.CONSTRAINTS},
    ConversationGoal.GEAR: {TasteField.PREFERENCES, TasteField.OWNED},
    ConversationGoal.DEEPEN: {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    },
}


def _active_signals(state: ConversationState):
    return state.profile.active_signals()


def _query_signals(state: ConversationState):
    return [signal for signal in _active_signals(state) if signal.link_role == LinkRole.QUERY]


def _taste_keys(state: ConversationState) -> set[tuple[str, ...]]:
    """지금까지 나온 취향의 축. 활성 한도 밖으로 밀린 취향도 센다. 축이 없으면 값 자체로 센다."""
    return {
        signal.taxonomy_path or (signal.normalized_value,)
        for signal in state.profile.signals
        if signal.field == TasteField.PREFERENCES and signal.status != SignalStatus.SUPERSEDED
    }


def taste_axis_count(state: ConversationState) -> int:
    return len(_taste_keys(state))


def uncovered_taste_axes(state: ConversationState) -> list[TasteAxis]:
    """대화로 물어볼 수 있는 축 중 아직 취향이 나오지 않은 것. 묻기 쉬운 순서로 돌려준다."""
    covered = {
        axis.key for key in _taste_keys(state) if (axis := taste_axis_for_path(key)) is not None
    }
    return [TASTE_AXES_BY_KEY[key] for key in ASK_PRIORITY if key not in covered]


def refresh_derived_coverage(state: ConversationState) -> None:
    query_signals = _query_signals(state)
    if len(query_signals) >= MIN_QUERY_SIGNALS:
        state.goal_coverage[GoalArea.INTEREST] = CoverageStatus.FOUND
    elif state.goal_coverage[GoalArea.INTEREST] == CoverageStatus.FOUND:
        state.goal_coverage[GoalArea.INTEREST] = CoverageStatus.PENDING

    if taste_axis_count(state) >= MIN_TASTE_AXES:
        state.goal_coverage[GoalArea.TASTE] = CoverageStatus.FOUND
    elif state.goal_coverage[GoalArea.TASTE] == CoverageStatus.FOUND:
        state.goal_coverage[GoalArea.TASTE] = CoverageStatus.PENDING

    high_confidence = [
        signal for signal in query_signals if signal.confidence >= HIGH_CONFIDENCE_THRESHOLD
    ]
    if len(high_confidence) >= MIN_HIGH_CONFIDENCE_QUERY_SIGNALS:
        state.goal_coverage[GoalArea.DEEPEN] = CoverageStatus.FOUND
    elif state.goal_coverage[GoalArea.DEEPEN] == CoverageStatus.FOUND:
        state.goal_coverage[GoalArea.DEEPEN] = CoverageStatus.PENDING

    active = _active_signals(state)
    if any(signal.field in {TasteField.PREFERENCES, TasteField.OWNED} for signal in active):
        state.goal_coverage[GoalArea.GEAR] = CoverageStatus.FOUND
    elif state.goal_coverage[GoalArea.GEAR] == CoverageStatus.FOUND:
        state.goal_coverage[GoalArea.GEAR] = CoverageStatus.PENDING

    if any(signal.field in {TasteField.DISLIKES, TasteField.CONSTRAINTS} for signal in active):
        state.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.FOUND
    elif state.goal_coverage[GoalArea.EXCLUSION] == CoverageStatus.FOUND:
        state.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.PENDING


def recommendation_readiness(state: ConversationState) -> ReadinessResult:
    refresh_derived_coverage(state)
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
    if state.goal_coverage[GoalArea.GEAR] not in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
    }:
        missing.append("gear")
    # companion은 싫어하는 것을 준비 조건에서 뺀다. 필수로 두면 별로였던 경험을 캐묻게 된다.
    if state.conversation_style != ConversationStyle.COMPANION and state.goal_coverage[
        GoalArea.EXCLUSION
    ] not in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
    }:
        missing.append("exclusion")
    return ReadinessResult(sufficient=not missing, missing_signals=tuple(missing))


def conversation_progress(state: ConversationState) -> int:
    if state.status in {
        SessionStatus.INPUT_LOCKED,
        SessionStatus.REVIEW,
        SessionStatus.CLOSED,
    }:
        return 100

    refresh_derived_coverage(state)
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
    if state.goal_coverage[GoalArea.GEAR] in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
    }:
        progress += GEAR_PROGRESS_WEIGHT
    if state.conversation_style == ConversationStyle.COMPANION:
        # 싫어하는 것을 빼고 100으로 맞추고, 한 번 오른 진행률은 내리지 않는다.
        progress = progress * 100 / (100 - EXCLUSION_PROGRESS_WEIGHT)
        return max(round(progress), state.progress_floor)
    if state.goal_coverage[GoalArea.EXCLUSION] in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
    }:
        progress += EXCLUSION_PROGRESS_WEIGHT
    return round(progress)


def fields_for_goal(goal: ConversationGoal) -> frozenset[TasteField]:
    return frozenset(_GOAL_FIELDS.get(goal, set()))


def _recent_user_messages(state: ConversationState) -> list[str]:
    return [turn.content.strip() for turn in state.history if turn.role == "user"][-3:]


def _shows_disengagement(state: ConversationState) -> bool:
    messages = _recent_user_messages(state)
    # 한국어 대화체는 짧은 답이 흔해서, 세 번 연속 짧게 줄어들 때만 이탈로 본다.
    return (
        len(messages) == 3
        and len(messages[0]) > len(messages[1]) > len(messages[2])
        and all(len(message) < DISENGAGED_MESSAGE_LENGTH for message in messages)
    )


def _area_available(state: ConversationState, area: GoalArea) -> bool:
    return state.goal_coverage[area] not in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
        CoverageStatus.EXHAUSTED,
    }


def decide_goal(state: ConversationState, utterance: str) -> GoalDecision:
    if state.conversation_style == ConversationStyle.REFLECTIVE:
        return _decide_reflective(state)
    if state.conversation_style == ConversationStyle.COMPANION:
        return _decide_companion(state, utterance)
    if state.status != SessionStatus.ACTIVE:
        return GoalDecision(
            ConversationGoal.WRAP,
            state.completion_reason or CompletionReason.USER_EXIT,
        )
    if state.turn_count >= MAX_TURNS:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.MAX_CYCLES)
    if _shows_disengagement(state):
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)
    if recommendation_readiness(state).sufficient:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.SUFFICIENT)

    query_count = len(_query_signals(state))
    interest_attempts = state.goal_attempts.get(ConversationGoal.INTEREST.value, 0)
    routine_attempts = state.goal_attempts.get(ConversationGoal.INTEREST_VIA_ROUTINE.value, 0)
    if query_count < MIN_QUERY_SIGNALS and _area_available(state, GoalArea.INTEREST):
        if interest_attempts < 2:
            return GoalDecision(ConversationGoal.INTEREST)
        if interest_attempts + routine_attempts < MAX_GOAL_ATTEMPTS:
            return GoalDecision(ConversationGoal.INTEREST_VIA_ROUTINE)

    # 관심사가 하나라도 나오면 그 관심사를 어떻게 즐기는지에서 취향 축을 찾는다.
    if (
        query_count >= 1
        and _area_available(state, GoalArea.TASTE)
        and state.goal_attempts.get(ConversationGoal.TASTE.value, 0) < MAX_GOAL_ATTEMPTS
    ):
        return GoalDecision(ConversationGoal.TASTE)
    if (
        (state.turn_count >= 3 or query_count >= MIN_QUERY_SIGNALS)
        and _area_available(state, GoalArea.EXCLUSION)
        and state.goal_attempts.get(ConversationGoal.DISLIKE.value, 0) < MAX_GOAL_ATTEMPTS
    ):
        return GoalDecision(ConversationGoal.DISLIKE)
    if (
        query_count >= 1
        and _area_available(state, GoalArea.GEAR)
        and state.goal_attempts.get(ConversationGoal.GEAR.value, 0) < MAX_GOAL_ATTEMPTS
    ):
        return GoalDecision(ConversationGoal.GEAR)
    if (
        query_count >= 1
        and _area_available(state, GoalArea.DEEPEN)
        and state.goal_attempts.get(ConversationGoal.DEEPEN.value, 0) < MAX_GOAL_ATTEMPTS
    ):
        return GoalDecision(ConversationGoal.DEEPEN)

    # 기본 순서를 다 돌았어도 완료 조건에 모자란 신호가 있으면 끝내지 않고 그 신호를 묻는다.
    # 대화는 MAX_TURNS, 이탈 판정, 또는 더 물을 게 없을 때 끝난다.
    readiness = recommendation_readiness(state)
    follow_up = _goal_for_missing(state, readiness.missing_signals, query_count)
    if follow_up is not None:
        return GoalDecision(follow_up)

    return GoalDecision(ConversationGoal.WRAP, CompletionReason.MAX_CYCLES)


def _goal_for_missing(
    state: ConversationState,
    missing: tuple[str, ...],
    query_count: int,
) -> ConversationGoal | None:
    """모자란 신호를 채울 목표 중 덜 쓴 것을 고른다. 직전 목표는 되도록 피한다."""
    candidates: list[ConversationGoal] = []
    if "query_signals" in missing:
        candidates += [ConversationGoal.INTEREST, ConversationGoal.INTEREST_VIA_ROUTINE]
    if query_count >= 1 and (
        "query_signals" in missing or "high_confidence_query_signals" in missing
    ):
        candidates.append(ConversationGoal.DEEPEN)
    if (
        "taste_axes" in missing
        and query_count >= 1
        and state.goal_attempts.get(ConversationGoal.TASTE.value, 0) < MAX_FOLLOW_UP_ATTEMPTS
    ):
        candidates.append(ConversationGoal.TASTE)
    # 소지품과 싫은 것은 계속 물으면 캐묻는 느낌이 들어서 횟수를 제한한다.
    if (
        "gear" in missing
        and query_count >= 1
        and state.goal_attempts.get(ConversationGoal.GEAR.value, 0) < MAX_FOLLOW_UP_ATTEMPTS
    ):
        candidates.append(ConversationGoal.GEAR)
    if (
        "exclusion" in missing
        and state.goal_attempts.get(ConversationGoal.DISLIKE.value, 0) < MAX_FOLLOW_UP_ATTEMPTS
    ):
        candidates.append(ConversationGoal.DISLIKE)
    if not candidates:
        return None
    if len(candidates) > 1 and state.last_goal in candidates:
        candidates.remove(state.last_goal)
    return min(candidates, key=lambda goal: state.goal_attempts.get(goal.value, 0))


def _decide_companion(state: ConversationState, utterance: str) -> GoalDecision:
    """사용자 에너지와 최근 흐름을 보고 이번 턴의 반응 방식(move)을 고른다.

    위에서부터 차례로 보고 처음 맞는 move를 고른다. 정보 공백은 move를 정하지 않고,
    건너가기나 놀이에서 어느 화제로 갈지에만 가중치를 준다.
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
            return chat(ConversationMove.PLAY, _pick_scene(state, readiness.missing_signals))
        return chat(ConversationMove.ADD)
    if state.question_streak >= COMPANION_MAX_QUESTION_STREAK:
        if _can_reflect_back(state, energy):
            return chat(ConversationMove.REFLECT_BACK)
        return chat(ConversationMove.ADD)
    if state.turns_since_new_interest >= COMPANION_STAY_TURNS and _energy_falling(state, energy):
        return chat(ConversationMove.BRIDGE, _pick_scene(state, readiness.missing_signals))
    if _can_reflect_back(state, energy):
        return chat(ConversationMove.REFLECT_BACK)
    return chat(ConversationMove.FOLLOW)


def _good_moment_to_close(state: ConversationState, energy: float | None) -> bool:
    """준비가 됐어도 즐겁게 이어지면 끝내지 않는다.

    오래 했거나 에너지가 꺾이기 시작할 때 끝낸다.
    """
    if state.turn_count >= COMPANION_CLOSE_AFTER_TURN:
        return True
    return energy is not None and state.peak_energy - energy >= COMPANION_CLOSE_ENERGY_DROP


def _energy_falling(state: ConversationState, energy: float | None) -> bool:
    if energy is None:
        return False
    return energy < ENERGY_FALLING_LEVEL or energy <= state.peak_energy - ENERGY_FALLING_DROP


def _can_reflect_back(state: ConversationState, energy: float | None) -> bool:
    if (
        energy is None
        or energy < REFLECT_BACK_MIN_ENERGY
        or state.reflect_back_count >= MAX_REFLECT_BACK
        or state.turn_count < REFLECT_BACK_FROM_TURN
        or ConversationMove.REFLECT_BACK.value in state.move_history[-4:]
    ):
        return False
    clues = reflection_clues(state)
    # 같은 단서로 두 번 되비추지 않는다.
    fresh = [signal for signal in clues if signal.value not in state.reflection_values]
    return len(clues) >= 2 and bool(fresh)


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


def _info_gap_weight(turn_count: int) -> float:
    """정보 공백을 화제 선택에 얼마나 세게 반영할지(λ). 초반에는 0이고 후반으로 갈수록 키운다."""
    if turn_count < 7:
        return 0.0
    if turn_count < 13:
        return 0.5
    return 1.0


def _pick_scene(state: ConversationState, missing: tuple[str, ...]) -> str | None:
    """아직 쓰지 않은 일상 장면 중 하나를 고른다.

    후반일수록 추천 공백을 채우는 장면을 앞세운다.
    """
    weight = _info_gap_weight(state.turn_count)
    offset = state.conversation_room_id % len(COMPANION_SCENES)
    rotated = COMPANION_SCENES[offset:] + COMPANION_SCENES[:offset]
    unused = [(scene, gaps) for scene, gaps in rotated if scene not in state.used_scenes]
    if not unused:
        return None
    # 같은 점수면 회전 순서를 따른다. 방마다 시작 장면이 달라서 대화가 매번 같아지지 않는다.
    best = max(unused, key=lambda item: weight * bool(item[1] & set(missing)))
    return best[0]


def record_companion_turn(
    state: ConversationState,
    *,
    utterance: str,
    reply: str,
    move: ConversationMove | None,
    scene: str | None,
    answer_depth: float | None,
    new_interests: list[str],
) -> None:
    """턴이 끝난 뒤 에너지와 move 흐름을 기록한다. companion 대화에서만 쓴다.

    history에 이번 발화를 넣기 전에 불러야 평균 길이가 이번 발화 없이 비교된다.
    """
    if state.conversation_style != ConversationStyle.COMPANION:
        return
    current = record_energy(state, utterance, answer_depth=answer_depth)
    if new_interests and current >= state.best_topic_energy:
        state.best_topic = new_interests[0]
        state.best_topic_energy = current
    state.question_streak = state.question_streak + 1 if "?" in reply else 0
    if move is None:
        return
    state.move_history = [*state.move_history, move.value][-MOVE_HISTORY_LIMIT:]
    if move == ConversationMove.RETURN and scene:
        state.returned_topics.append(scene)
    if move in {ConversationMove.BRIDGE, ConversationMove.PLAY} and scene:
        state.used_scenes.append(scene)
    if move == ConversationMove.REFLECT_BACK:
        state.reflect_back_count += 1


def known_query_values(state: ConversationState) -> set[str]:
    return {
        normalize_text(signal.value)
        for signal in state.profile.signals
        if signal.link_role == LinkRole.QUERY
    }


def record_interest_progress(
    state: ConversationState,
    known_values: set[str],
    accepted: tuple[ProfileSignal, ...],
) -> None:
    """이번 턴에 처음 보는 관심사가 들어왔는지로 같은 화제에 머문 턴 수를 센다.

    취향(preferences)은 기존 관심사를 더 자세히 들은 것이라 새 관심사로 치지 않는다.
    """
    found_new = any(
        signal.link_role == LinkRole.QUERY and normalize_text(signal.value) not in known_values
        for signal in accepted
    )
    state.turns_since_new_interest = 0 if found_new else state.turns_since_new_interest + 1


def should_broaden_topic(state: ConversationState) -> bool:
    return state.turns_since_new_interest >= TOPIC_BROADEN_AFTER_TURNS


def record_goal_attempt(state: ConversationState, goal: ConversationGoal) -> None:
    if goal in {ConversationGoal.OPENING, ConversationGoal.WRAP}:
        return
    state.goal_attempts[goal.value] = state.goal_attempts.get(goal.value, 0) + 1
    state.last_goal = goal


def apply_goal_assessment(
    state: ConversationState,
    goal: ConversationGoal,
    assessment: AssessmentStatus,
) -> None:
    area = _GOAL_AREA.get(goal)
    if area is None:
        return

    can_resolve_directly = area in {GoalArea.GEAR, GoalArea.EXCLUSION}
    if assessment == AssessmentStatus.FOUND and can_resolve_directly:
        state.goal_coverage[area] = CoverageStatus.FOUND
    elif assessment == AssessmentStatus.CONFIRMED_NONE and can_resolve_directly:
        state.goal_coverage[area] = CoverageStatus.CONFIRMED_NONE
    else:
        state.goal_coverage[area] = CoverageStatus.UNRESOLVED
        if area == GoalArea.INTEREST:
            attempts = state.goal_attempts.get(ConversationGoal.INTEREST.value, 0)
            attempts += state.goal_attempts.get(ConversationGoal.INTEREST_VIA_ROUTINE.value, 0)
        else:
            attempts = state.goal_attempts.get(goal.value, 0)
        if attempts >= MAX_GOAL_ATTEMPTS:
            state.goal_coverage[area] = CoverageStatus.EXHAUSTED
    refresh_derived_coverage(state)


# ---- reflective 대화 ----


def _decide_reflective(state: ConversationState) -> GoalDecision:
    """이야기 줄기의 단계로 다음 질문을 정한다. 부족한 정보는 줄기 안의 단계로 채운다."""
    if state.status != SessionStatus.ACTIVE:
        return GoalDecision(
            ConversationGoal.WRAP,
            state.completion_reason or CompletionReason.USER_EXIT,
        )
    if state.turn_count >= MAX_TURNS:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.MAX_CYCLES)
    if state.shallow_streak >= SHALLOW_STREAK_TO_WRAP or _shows_disengagement(state):
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.USER_EXIT)

    ready = recommendation_readiness(state).sufficient
    near_end = state.turn_count >= MAX_TURNS - 2
    # 되비추기 질문에 답이 오면 마무리한다. 정정으로 준비가 모자라졌으면 이야기를 이어간다.
    if state.last_goal == ConversationGoal.REFLECT and (ready or near_end):
        reason = CompletionReason.SUFFICIENT if ready else CompletionReason.MAX_CYCLES
        return GoalDecision(ConversationGoal.WRAP, reason)
    if not state.reflected and (ready or near_end) and reflection_clues(state):
        return GoalDecision(ConversationGoal.REFLECT)
    if ready:
        return GoalDecision(ConversationGoal.WRAP, CompletionReason.SUFFICIENT)
    return GoalDecision(_STAGE_GOAL[state.thread_stage])


def reflection_clues(state: ConversationState) -> list[ProfileSignal]:
    """되비추기에 쓸 단서. 단서가 가장 많이 쌓인 묶음 몇 개에서 고른다."""
    groups = friend_signal_groups(state.profile, _REFLECTION_FIELDS, limit=REFLECTION_CLUES * 2)
    clues = [signal for _, signals in groups[:REFLECTION_GROUPS] for signal in signals]
    return clues[:REFLECTION_CLUES]


def latest_motive(state: ConversationState) -> ProfileSignal | None:
    """가장 최근에 드러난 동기·가치. 화제를 옮길 때 다리로 쓴다."""
    motives = [
        signal
        for signal in state.profile.stored_signals()
        if signal.field == TasteField.PREFERENCES and signal.aspect == PreferenceAspect.MOTIVE
    ]
    return max(motives, key=lambda signal: signal.updated_at, default=None)


def _topic_taste_axes(state: ConversationState) -> int:
    if state.thread_topic is None:
        return 0
    topic = normalize_text(state.thread_topic)
    return len(
        {
            signal.taxonomy_path or (signal.normalized_value,)
            for signal in state.profile.stored_signals()
            if signal.field == TasteField.PREFERENCES
            and signal.target
            and normalize_text(signal.target) == topic
        }
    )


def _topic_has_motive(state: ConversationState) -> bool:
    if state.thread_topic is None:
        return False
    topic = normalize_text(state.thread_topic)
    return any(
        signal.aspect == PreferenceAspect.MOTIVE
        and signal.target
        and normalize_text(signal.target) == topic
        for signal in state.profile.stored_signals()
    )


def _next_stage(state: ConversationState, current: ThreadStage) -> ThreadStage:
    start = 0 if current == ThreadStage.WHAT else _STAGE_ORDER.index(current) + 1
    for stage in _STAGE_ORDER[start:]:
        # 이 관심사를 어떻게 즐기는지 이미 두 축 이상 들었으면 건너뛴다.
        if stage == ThreadStage.HOW and _topic_taste_axes(state) >= MIN_TASTE_AXES:
            continue
        # 이 관심사에서 무엇을 얻는지 이미 들었거나 "왜"를 충분히 물었으면 건너뛴다.
        if stage == ThreadStage.WHY and (
            _topic_has_motive(state) or state.why_count >= MAX_WHY_QUESTIONS
        ):
            continue
        if stage == ThreadStage.CONTRAST and not _area_available(state, GoalArea.EXCLUSION):
            continue
        return stage
    return ThreadStage.BRIDGE


def advance_thread(
    state: ConversationState,
    goal: ConversationGoal,
    *,
    new_interests: list[str],
    answer_depth: float | None,
) -> None:
    """턴이 끝난 뒤 이야기 줄기를 다음 단계로 옮긴다. reflective 대화에서만 쓴다."""
    if state.conversation_style != ConversationStyle.REFLECTIVE:
        return
    refresh_derived_coverage(state)
    shallow = answer_depth is not None and answer_depth < SHALLOW_ANSWER_DEPTH
    if answer_depth is not None:
        state.last_answer_depth = answer_depth
        state.shallow_streak = state.shallow_streak + 1 if shallow else 0
    if goal == ConversationGoal.REFLECT:
        state.reflected = True
        return
    if goal in {ConversationGoal.OPENING, ConversationGoal.WRAP}:
        return
    if goal == ConversationGoal.DEEPEN:
        state.why_count += 1

    # 줄기가 없거나 화제를 옮기는 중이면, 새 관심사가 나왔을 때 그걸로 새 줄기를 시작한다.
    if state.thread_topic is None or state.thread_stage in {ThreadStage.WHAT, ThreadStage.BRIDGE}:
        if new_interests:
            state.thread_topic = new_interests[0]
            state.thread_turns = 0
            state.thread_stage = _next_stage(state, ThreadStage.WHAT)
        else:
            state.thread_stage = ThreadStage.WHAT
        return

    state.thread_turns += 1
    # 답이 짧아졌거나 한 줄기에 오래 머물렀으면 가볍게 다른 영역으로 건너간다.
    if shallow or state.thread_turns >= MAX_THREAD_TURNS:
        state.thread_stage = ThreadStage.BRIDGE
        return
    state.thread_stage = _next_stage(state, state.thread_stage)
