from __future__ import annotations

from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    CoverageStatus,
    GoalArea,
    GoalDecision,
    ReadinessResult,
    SessionStatus,
)
from app.domain.profile.models import AssessmentStatus, LinkRole, TasteField

MAX_TURNS = 20
MIN_QUERY_SIGNALS = 3
MIN_HIGH_CONFIDENCE_QUERY_SIGNALS = 2
HIGH_CONFIDENCE_THRESHOLD = 0.6
MAX_GOAL_ATTEMPTS = 3

_GOAL_AREA = {
    ConversationGoal.INTEREST: GoalArea.INTEREST,
    ConversationGoal.INTEREST_VIA_ROUTINE: GoalArea.INTEREST,
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
    ConversationGoal.DISLIKE: {TasteField.DISLIKES, TasteField.CONSTRAINTS},
    ConversationGoal.GEAR: {TasteField.PREFERENCES, TasteField.OWNED},
    ConversationGoal.DEEPEN: {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    },
    ConversationGoal.CORRECT: set(TasteField),
}


def _active_signals(state: ConversationState):
    return state.profile.active_signals()


def _query_signals(state: ConversationState):
    return [signal for signal in _active_signals(state) if signal.link_role == LinkRole.QUERY]


def refresh_derived_coverage(state: ConversationState) -> None:
    query_signals = _query_signals(state)
    if len(query_signals) >= MIN_QUERY_SIGNALS:
        state.goal_coverage[GoalArea.INTEREST] = CoverageStatus.FOUND
    elif state.goal_coverage[GoalArea.INTEREST] == CoverageStatus.FOUND:
        state.goal_coverage[GoalArea.INTEREST] = CoverageStatus.PENDING

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
    if state.goal_coverage[GoalArea.GEAR] not in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
    }:
        missing.append("gear")
    if state.goal_coverage[GoalArea.EXCLUSION] not in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
    }:
        missing.append("exclusion")
    return ReadinessResult(sufficient=not missing, missing_signals=tuple(missing))


def fields_for_goal(goal: ConversationGoal) -> frozenset[TasteField]:
    return frozenset(_GOAL_FIELDS.get(goal, set()))


def _recent_user_messages(state: ConversationState) -> list[str]:
    return [turn.content.strip() for turn in state.history if turn.role == "user"][-3:]


def _shows_disengagement(state: ConversationState) -> bool:
    messages = _recent_user_messages(state)
    return (
        len(messages) == 3
        and len(messages[0]) > len(messages[1]) > len(messages[2])
        and len(messages[2]) < 12
    )


def _area_available(state: ConversationState, area: GoalArea) -> bool:
    return state.goal_coverage[area] not in {
        CoverageStatus.FOUND,
        CoverageStatus.CONFIRMED_NONE,
        CoverageStatus.EXHAUSTED,
    }


def decide_goal(state: ConversationState, utterance: str) -> GoalDecision:
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

    return GoalDecision(ConversationGoal.WRAP, CompletionReason.MAX_CYCLES)


def record_goal_attempt(state: ConversationState, goal: ConversationGoal) -> None:
    if goal in {ConversationGoal.OPENING, ConversationGoal.CORRECT, ConversationGoal.WRAP}:
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
