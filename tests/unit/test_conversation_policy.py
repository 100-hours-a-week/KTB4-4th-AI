from datetime import UTC, datetime

from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    CoverageStatus,
    GoalArea,
)
from app.domain.conversation.policy import (
    apply_goal_assessment,
    decide_goal,
    recommendation_readiness,
)
from app.domain.profile.models import (
    AssessmentStatus,
    EvidenceType,
    IntentType,
    LinkRole,
    ProfileSignal,
    SignalStatus,
    TasteField,
    Visibility,
)


def signal(field: TasteField, value: str, confidence: float = 0.9) -> ProfileSignal:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=confidence,
        link_role=(
            LinkRole.QUERY
            if field
            in {
                TasteField.INTERESTS,
                TasteField.HOBBIES,
                TasteField.WANTS,
                TasteField.UNAFFORDABLE,
                TasteField.CONSUMABLES,
            }
            else LinkRole.WEIGHT
        ),
        visibility=Visibility.FRIENDS,
        intent_type=IntentType.BOTH,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=now,
        updated_at=now,
        status=SignalStatus.ACTIVE,
    )


def state() -> ConversationState:
    return ConversationState(user_id=1, conversation_room_id=101)


def test_interest_is_first_goal_when_query_signals_are_missing() -> None:
    decision = decide_goal(state(), "주말에는 뭘 할까요")

    assert decision.goal == ConversationGoal.INTEREST


def test_natural_language_exit_words_do_not_force_session_close() -> None:
    conversation = state()

    assert decide_goal(conversation, "그만").goal == ConversationGoal.INTEREST
    assert decide_goal(conversation, "요즘 수영을 그만뒀어요").goal == ConversationGoal.INTEREST


def test_interest_switches_to_routine_after_two_unsuccessful_attempts() -> None:
    conversation = state()
    conversation.goal_attempts[ConversationGoal.INTEREST.value] = 2

    decision = decide_goal(conversation, "잘 모르겠어요")

    assert decision.goal == ConversationGoal.INTEREST_VIA_ROUTINE


def test_ready_profile_wraps_before_eight_turns() -> None:
    conversation = state()
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑"),
            signal(TasteField.HOBBIES, "핸드드립"),
            signal(TasteField.WANTS, "가벼운 컵"),
        ]
    )
    conversation.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE
    conversation.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.CONFIRMED_NONE

    readiness = recommendation_readiness(conversation)
    decision = decide_goal(conversation, "좋아요")

    assert readiness.sufficient is True
    assert conversation.turn_count < 8
    assert decision.goal == ConversationGoal.WRAP
    assert decision.completion_reason == CompletionReason.SUFFICIENT


def test_exclusion_is_asked_early_when_three_query_signals_are_already_known() -> None:
    conversation = state()
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑"),
            signal(TasteField.HOBBIES, "핸드드립"),
            signal(TasteField.WANTS, "가벼운 컵"),
        ]
    )

    decision = decide_goal(conversation, "사진도 좋아해요")

    assert conversation.turn_count == 0
    assert decision.goal == ConversationGoal.DISLIKE


def test_empty_arrays_do_not_mean_goal_was_resolved() -> None:
    conversation = state()
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑"),
            signal(TasteField.HOBBIES, "핸드드립"),
            signal(TasteField.WANTS, "가벼운 컵"),
        ]
    )

    readiness = recommendation_readiness(conversation)

    assert readiness.sufficient is False
    assert "gear" in readiness.missing_signals
    assert "exclusion" in readiness.missing_signals


def test_one_interest_does_not_complete_three_signal_interest_target() -> None:
    conversation = state()
    conversation.profile.signals.append(signal(TasteField.INTERESTS, "캠핑"))

    apply_goal_assessment(
        conversation,
        ConversationGoal.INTEREST,
        AssessmentStatus.FOUND,
    )

    assert conversation.goal_coverage[GoalArea.INTEREST] == CoverageStatus.UNRESOLVED
    assert decide_goal(conversation, "좋아요").goal == ConversationGoal.INTEREST


def test_confirmed_none_only_resolves_optional_gear_and_exclusion_goals() -> None:
    conversation = state()

    apply_goal_assessment(
        conversation,
        ConversationGoal.GEAR,
        AssessmentStatus.CONFIRMED_NONE,
    )
    apply_goal_assessment(
        conversation,
        ConversationGoal.DISLIKE,
        AssessmentStatus.CONFIRMED_NONE,
    )
    apply_goal_assessment(
        conversation,
        ConversationGoal.INTEREST,
        AssessmentStatus.CONFIRMED_NONE,
    )

    assert conversation.goal_coverage[GoalArea.GEAR] == CoverageStatus.CONFIRMED_NONE
    assert conversation.goal_coverage[GoalArea.EXCLUSION] == CoverageStatus.CONFIRMED_NONE
    assert conversation.goal_coverage[GoalArea.INTEREST] == CoverageStatus.UNRESOLVED


def test_readiness_is_recomputed_when_last_active_signal_is_removed() -> None:
    conversation = state()
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑"),
            signal(TasteField.HOBBIES, "핸드드립"),
            signal(TasteField.WANTS, "가벼운 컵"),
        ]
    )
    exclusion = signal(TasteField.DISLIKES, "강한 향")
    exclusion.link_role = LinkRole.FILTER
    conversation.profile.signals.append(exclusion)
    conversation.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE

    assert recommendation_readiness(conversation).sufficient is True

    exclusion.status = SignalStatus.SUPERSEDED
    readiness = recommendation_readiness(conversation)

    assert readiness.sufficient is False
    assert "exclusion" in readiness.missing_signals
    assert conversation.goal_coverage[GoalArea.EXCLUSION] == CoverageStatus.PENDING
