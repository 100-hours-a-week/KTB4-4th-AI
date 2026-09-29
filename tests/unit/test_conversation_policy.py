from datetime import UTC, datetime

from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    ConversationTurn,
    CoverageStatus,
    GoalArea,
    SessionStatus,
)
from app.domain.conversation.policy import (
    apply_goal_assessment,
    conversation_progress,
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
            signal(TasteField.INTERESTS, "사진"),
            signal(TasteField.HOBBIES, "러닝"),
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


def test_exclusion_is_asked_early_when_five_query_signals_are_already_known() -> None:
    conversation = state()
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑"),
            signal(TasteField.HOBBIES, "핸드드립"),
            signal(TasteField.WANTS, "가벼운 컵"),
            signal(TasteField.INTERESTS, "사진"),
            signal(TasteField.HOBBIES, "러닝"),
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
            signal(TasteField.INTERESTS, "사진"),
            signal(TasteField.HOBBIES, "러닝"),
        ]
    )

    readiness = recommendation_readiness(conversation)

    assert readiness.sufficient is False
    assert "gear" in readiness.missing_signals
    assert "exclusion" in readiness.missing_signals


def test_one_interest_does_not_complete_five_signal_interest_target() -> None:
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
            signal(TasteField.INTERESTS, "사진"),
            signal(TasteField.HOBBIES, "러닝"),
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


def test_progress_uses_extracted_signals_and_resolved_goals() -> None:
    conversation = state()
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑", confidence=0.9),
            signal(TasteField.HOBBIES, "핸드드립", confidence=0.5),
        ]
    )
    conversation.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE

    assert conversation_progress(conversation) == 47


def test_progress_is_complete_when_input_is_locked_or_under_review() -> None:
    conversation = state()
    conversation.status = SessionStatus.INPUT_LOCKED
    assert conversation_progress(conversation) == 100

    conversation.status = SessionStatus.REVIEW
    assert conversation_progress(conversation) == 100


def test_goals_running_out_below_five_tastes_keeps_exploring_instead_of_wrapping() -> None:
    conversation = state()
    conversation.turn_count = 10
    conversation.profile.signals.extend(
        [signal(TasteField.INTERESTS, "캠핑"), signal(TasteField.HOBBIES, "핸드드립")]
    )
    for goal in (
        ConversationGoal.INTEREST,
        ConversationGoal.INTEREST_VIA_ROUTINE,
        ConversationGoal.DISLIKE,
        ConversationGoal.GEAR,
        ConversationGoal.DEEPEN,
    ):
        conversation.goal_attempts[goal.value] = 3
    conversation.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE
    conversation.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.CONFIRMED_NONE
    conversation.last_goal = ConversationGoal.DEEPEN

    decision = decide_goal(conversation, "캠핑 얘기 좀 더 할래요")

    assert decision.goal in {ConversationGoal.INTEREST, ConversationGoal.INTEREST_VIA_ROUTINE}
    assert decision.completion_reason is None


def test_short_but_not_tiny_answers_are_not_treated_as_disengagement() -> None:
    conversation = state()
    for text in ("주말에 친구랑 캠핑 갔다 왔어", "고기 구워 먹었어", "재밌었어"):
        conversation.history.append(
            ConversationTurn(role="user", content=text, created_at=datetime(2026, 9, 18))
        )

    assert decide_goal(conversation, "재밌었어").goal != ConversationGoal.WRAP


def test_missing_exclusion_is_asked_again_after_default_order_runs_out() -> None:
    conversation = state()
    conversation.turn_count = 12
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑"),
            signal(TasteField.HOBBIES, "핸드드립"),
            signal(TasteField.WANTS, "가벼운 컵"),
            signal(TasteField.INTERESTS, "사진"),
            signal(TasteField.HOBBIES, "러닝"),
        ]
    )
    for goal in (ConversationGoal.DISLIKE, ConversationGoal.GEAR, ConversationGoal.DEEPEN):
        conversation.goal_attempts[goal.value] = 3
    conversation.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE
    conversation.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.EXHAUSTED

    assert decide_goal(conversation, "음").goal == ConversationGoal.DISLIKE

    conversation.goal_attempts[ConversationGoal.DISLIKE.value] = 5
    decision = decide_goal(conversation, "음")
    assert decision.goal == ConversationGoal.WRAP
    assert decision.completion_reason == CompletionReason.MAX_CYCLES
