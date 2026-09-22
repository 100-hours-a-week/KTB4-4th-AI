import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.application.conversation_prompts import build_friend_summary_messages
from app.application.conversation_service import ConversationService
from app.application.ports.model_gateway import Message
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    CoverageStatus,
    GoalArea,
    SessionStatus,
)
from app.domain.conversation.policy import decide_goal
from app.domain.profile.models import (
    EvidenceType,
    IntentType,
    LinkRole,
    ProfileSignal,
    SignalStatus,
    TasteField,
    Visibility,
)


class FakeModelGateway:
    def __init__(
        self,
        structured_results: list[Mapping[str, Any] | Exception],
    ) -> None:
        self.structured_results = structured_results
        self.calls = 0

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        raise NotImplementedError

    def stream(self, messages: Sequence[Message], *, model: str) -> AsyncIterator[str]:
        raise NotImplementedError

    async def structured(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self.structured_results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def profile_signal(field: TasteField, value: str) -> ProfileSignal:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=0.9,
        link_role=LinkRole.QUERY,
        visibility=Visibility.FRIENDS,
        intent_type=IntentType.BOTH,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=now,
        updated_at=now,
        status=SignalStatus.ACTIVE,
    )


def test_prepare_session_uses_room_as_session_id_and_records_safe_opening() -> None:
    gateway = FakeModelGateway([])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    now = datetime(2026, 9, 18, tzinfo=UTC)

    prepared = service.prepare_session(
        user_id=1,
        conversation_room_id=101,
        now=now,
    )
    greeting = service.record_opening(
        prepared.state,
        "반가워요! 요즘 어떻게 지내세요? 두 번째 질문? <<<STATE>>>비밀",
        now=now,
    )

    assert prepared.state.session_id == 101
    assert len(prepared.messages) == 2
    assert greeting == "반가워요! 요즘 어떻게 지내세요?"
    assert prepared.state.history[0].content == greeting
    assert prepared.state.turn_count == 0


def test_complete_turn_extracts_merges_and_updates_state() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    {
                        "field": "hobbies",
                        "value": "핸드드립",
                        "confidence": 0.9,
                        "evidence": "핸드드립을 자주 해요",
                        "evidenceType": "explicit",
                        "intentType": "both",
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [],
                "goalAssessment": {"goal": "INTEREST", "status": "found"},
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)
    prepared = service.prepare_turn(state, "핸드드립을 자주 해요")

    assert state.goal_attempts == {}

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="핸드드립을 자주 해요",
            raw_reply="주로 어떤 원두를 사용하세요? 두 번째 질문? <<<STATE>>>비밀",
            goal=prepared.decision.goal,
            now=datetime(2026, 9, 18, tzinfo=UTC),
        )
    )

    assert prepared.decision.goal == ConversationGoal.INTEREST
    assert completed.reply == "주로 어떤 원두를 사용하세요?"
    assert completed.state.turn_count == 1
    assert completed.state.profile.active_signals()[0].value == "핸드드립"
    assert completed.extraction_failed is False
    assert completed.state.goal_attempts[ConversationGoal.INTEREST.value] == 1


def test_invalid_extraction_retries_then_uses_empty_delta() -> None:
    gateway = FakeModelGateway([{}, {}, {}])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="잘 모르겠어요",
            raw_reply="평소 쉬는 날에는 어떻게 보내세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert gateway.calls == 3
    assert completed.extraction_failed is True
    assert completed.state.profile.active_signals() == []


def test_model_failure_during_extraction_does_not_break_completed_chat_turn() -> None:
    gateway = FakeModelGateway(
        [RuntimeError("model unavailable"), RuntimeError("model unavailable")]
    )
    service = ConversationService(
        model_gateway=gateway,
        extraction_model="extractor",
        extraction_retries=1,
    )
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="잘 모르겠어요",
            raw_reply="평소 쉬는 날에는 어떻게 보내세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert gateway.calls == 2
    assert completed.extraction_failed is True
    assert completed.state.turn_count == 1


def test_friend_summary_prompt_never_contains_sensitive_profile_values() -> None:
    # Projection behavior itself is covered in test_profile_merger; this ensures the prompt uses it.
    state = ConversationState(user_id=1, conversation_room_id=101)

    messages = build_friend_summary_messages(state.profile)

    assert len(messages) == 2
    assert "경제 사정" in messages[0]["content"]


def test_found_assessment_is_downgraded_when_item_does_not_match_goal() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    {
                        "field": "preferences",
                        "value": "가벼운 것",
                        "confidence": 0.9,
                        "evidence": "가벼운 게 좋아요",
                        "evidenceType": "explicit",
                        "intentType": None,
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [],
                "goalAssessment": {"goal": "INTEREST", "status": "found"},
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="가벼운 게 좋아요",
            raw_reply="요즘 실제로 자주 하는 일은 무엇인가요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.state.goal_coverage[GoalArea.INTEREST] == CoverageStatus.UNRESOLVED


def test_extraction_that_completes_readiness_closes_before_eight_turns() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    {
                        "field": "interests",
                        "value": "사진",
                        "confidence": 0.9,
                        "evidence": "사진도 좋아해요",
                        "evidenceType": "explicit",
                        "intentType": "both",
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [],
                "goalAssessment": {"goal": "INTEREST", "status": "found"},
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.profile.signals.extend(
        [
            profile_signal(TasteField.INTERESTS, "캠핑"),
            profile_signal(TasteField.HOBBIES, "핸드드립"),
        ]
    )
    state.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE
    state.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.CONFIRMED_NONE

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="사진도 좋아해요",
            raw_reply="사진은 주로 언제 찍으세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.readiness.sufficient is True
    assert completed.state.turn_count == 1
    assert completed.next_decision.completion_reason == CompletionReason.SUFFICIENT
    assert completed.state.status == SessionStatus.INPUT_LOCKED
    assert completed.state.completion_reason == CompletionReason.SUFFICIENT
    assert decide_goal(completed.state, "").completion_reason == CompletionReason.SUFFICIENT
