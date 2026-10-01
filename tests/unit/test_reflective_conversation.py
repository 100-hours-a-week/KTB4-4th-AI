import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.application.conversation_prompts import build_opening_messages, build_reply_messages
from app.application.conversation_service import ConversationService
from app.application.ports.model_gateway import Message
from app.application.taste_judgment import (
    Candidate,
    CandidateKind,
    build_judgment_request,
    correction_targets,
    judgment_delta,
)
from app.domain.conversation.guards import CONCISE_REPLY_CHARS, sanitize_response
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    ConversationStyle,
    ConversationTurn,
    CoverageStatus,
    GoalArea,
    SessionStatus,
    ThreadStage,
)
from app.domain.conversation.policy import (
    MAX_THREAD_TURNS,
    advance_thread,
    decide_goal,
)
from app.domain.profile.models import (
    EvidenceType,
    IntentType,
    LinkRole,
    PreferenceAspect,
    ProfileSignal,
    SignalStatus,
    TasteField,
    Visibility,
)

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def signal(
    field: TasteField,
    value: str,
    *,
    target: str | None = None,
    path: tuple[str, ...] | None = None,
    aspect: PreferenceAspect | None = None,
) -> ProfileSignal:
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=0.9,
        link_role=(
            LinkRole.WEIGHT
            if field in {TasteField.PREFERENCES, TasteField.LIFESTYLE}
            else LinkRole.FILTER
            if field in {TasteField.DISLIKES, TasteField.CONSTRAINTS, TasteField.OWNED}
            else LinkRole.QUERY
        ),
        visibility=Visibility.FRIENDS,
        intent_type=IntentType.BOTH,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=NOW,
        updated_at=NOW,
        status=SignalStatus.ACTIVE,
        target=target,
        taxonomy_path=path,
        aspect=aspect,
    )


def reflective_state() -> ConversationState:
    return ConversationState(
        user_id=1,
        conversation_room_id=1,
        conversation_style=ConversationStyle.REFLECTIVE,
    )


def advance(
    state: ConversationState,
    goal: ConversationGoal,
    *,
    new: list[str] | None = None,
    depth: float | None = 1.5,
) -> None:
    advance_thread(state, goal, new_interests=new or [], answer_depth=depth)


def test_thread_moves_from_what_through_how_why_contrast_to_bridge() -> None:
    state = reflective_state()

    assert decide_goal(state, "").goal == ConversationGoal.INTEREST
    advance(state, ConversationGoal.INTEREST, new=["캠핑"])
    assert (state.thread_topic, state.thread_stage) == ("캠핑", ThreadStage.HOW)
    assert decide_goal(state, "").goal == ConversationGoal.TASTE

    advance(state, ConversationGoal.TASTE)
    assert state.thread_stage == ThreadStage.WHY
    advance(state, ConversationGoal.DEEPEN)
    assert state.why_count == 1
    assert state.thread_stage == ThreadStage.CONTRAST
    assert decide_goal(state, "").goal == ConversationGoal.DISLIKE

    advance(state, ConversationGoal.DISLIKE)
    assert state.thread_stage == ThreadStage.BRIDGE
    assert decide_goal(state, "").goal == ConversationGoal.BRIDGE

    advance(state, ConversationGoal.BRIDGE, new=["영화"])
    assert (state.thread_topic, state.thread_stage) == ("영화", ThreadStage.HOW)


def test_stages_are_skipped_when_already_answered() -> None:
    state = reflective_state()
    state.profile.signals.extend(
        [
            signal(
                TasteField.PREFERENCES,
                "혼자 가는 캠핑",
                target="캠핑",
                path=("취향", "사회", "인원"),
            ),
            signal(
                TasteField.PREFERENCES,
                "사람 없는 계곡",
                target="캠핑",
                path=("취향", "장소", "밀집도"),
            ),
            signal(TasteField.DISLIKES, "시끄러운 캠핑장"),
        ]
    )

    # 캠핑을 어떻게 즐기는지 이미 두 축을 들었고, 캠핑에서 얻는 것도 들었다. 싫은 것도 이미 있다.
    state.profile.signals.append(
        signal(
            TasteField.PREFERENCES,
            "아무 생각 안 하는 시간",
            target="캠핑",
            path=("취향", "동기", "휴식·재충전"),
            aspect=PreferenceAspect.MOTIVE,
        )
    )
    advance(state, ConversationGoal.INTEREST, new=["캠핑"])

    assert state.thread_stage == ThreadStage.BRIDGE


def test_shallow_answer_or_long_thread_moves_to_bridge() -> None:
    state = reflective_state()
    advance(state, ConversationGoal.INTEREST, new=["캠핑"])
    advance(state, ConversationGoal.TASTE, depth=0.2)

    assert state.thread_stage == ThreadStage.BRIDGE
    assert state.shallow_streak == 1

    state = reflective_state()
    advance(state, ConversationGoal.INTEREST, new=["캠핑"])
    state.thread_turns = MAX_THREAD_TURNS - 1
    advance(state, ConversationGoal.TASTE)
    assert state.thread_stage == ThreadStage.BRIDGE


def test_bridge_without_new_interest_returns_to_light_what_question() -> None:
    state = reflective_state()
    state.thread_topic = "캠핑"
    state.thread_stage = ThreadStage.BRIDGE

    advance(state, ConversationGoal.BRIDGE)

    assert state.thread_stage == ThreadStage.WHAT
    assert decide_goal(state, "").goal == ConversationGoal.INTEREST


def test_repeated_shallow_answers_end_the_conversation() -> None:
    state = reflective_state()
    for _ in range(3):
        advance(state, ConversationGoal.INTEREST, depth=0.1)

    decision = decide_goal(state, "")
    assert decision.goal == ConversationGoal.WRAP
    assert decision.completion_reason == CompletionReason.USER_EXIT


def _ready_profile(state: ConversationState) -> None:
    state.profile.signals.extend(
        [
            signal(TasteField.HOBBIES, "캠핑", path=("관심사", "아웃도어")),
            signal(TasteField.HOBBIES, "등산", path=("관심사", "아웃도어")),
            signal(TasteField.INTERESTS, "재즈", path=("관심사", "음악")),
            signal(TasteField.INTERESTS, "커피", path=("관심사", "음식·미식")),
            signal(TasteField.HOBBIES, "뜨개질", path=("관심사", "창작")),
            signal(
                TasteField.PREFERENCES,
                "혼자 가는 캠핑",
                target="캠핑",
                path=("취향", "사회", "인원"),
            ),
            signal(
                TasteField.PREFERENCES,
                "혼자 듣는 재즈",
                target="재즈",
                path=("취향", "사회", "인원"),
            ),
            signal(TasteField.PREFERENCES, "사람 없는 계곡", path=("취향", "장소", "밀집도")),
        ]
    )
    state.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.CONFIRMED_NONE


def test_reflects_once_before_wrapping_when_ready() -> None:
    state = reflective_state()
    state.turn_count = 6
    _ready_profile(state)

    assert decide_goal(state, "").goal == ConversationGoal.REFLECT

    state.last_goal = ConversationGoal.REFLECT
    state.reflected = True
    decision = decide_goal(state, "")
    assert decision.goal == ConversationGoal.WRAP
    assert decision.completion_reason == CompletionReason.SUFFICIENT


class FakeGateway:
    def __init__(self, structured_results: list[Mapping[str, Any]]) -> None:
        self.structured_results = structured_results

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        raise NotImplementedError

    def stream(self, messages: Sequence[Message], *, model: str) -> AsyncIterator[str]:
        raise NotImplementedError

    async def structured(self, messages, *, model, json_schema) -> Mapping[str, Any]:
        return self.structured_results.pop(0)


def test_reflect_turn_waits_for_answer_then_wraps_on_next_turn() -> None:
    service = ConversationService(
        model_gateway=FakeGateway([{"items": []}, {"items": []}]),
        extraction_model="extractor",
    )
    state = reflective_state()
    state.turn_count = 6
    _ready_profile(state)

    reflected = asyncio.run(
        service.complete_turn(
            state,
            utterance="요즘은 그 정도예요",
            raw_reply="혼자 조용히 보내는 시간을 아끼시는 것 같은데, 맞아요?",
            goal=ConversationGoal.REFLECT,
        )
    )

    assert reflected.state.status == SessionStatus.ACTIVE
    assert reflected.reply.endswith("맞아요?")
    assert reflected.state.reflected is True
    assert "혼자 가는 캠핑" in reflected.state.reflection_values
    assert reflected.next_decision.goal == ConversationGoal.WRAP

    closed = asyncio.run(
        service.complete_turn(
            state,
            utterance="네 맞아요",
            raw_reply="이야기 나눠 줘서 고마워요.",
            goal=ConversationGoal.WRAP,
            completion_reason=CompletionReason.SUFFICIENT,
        )
    )
    assert closed.state.status == SessionStatus.INPUT_LOCKED


def test_reflection_values_become_first_correction_targets() -> None:
    state = reflective_state()
    _ready_profile(state)
    state.last_goal = ConversationGoal.REFLECT
    state.reflection_values = ["사람 없는 계곡"]

    targets = correction_targets(state, "아뇨 그건 좀 아닌 것 같아요")

    assert targets[0].value == "사람 없는 계곡"


def test_concise_guard_keeps_only_the_question_when_reply_is_long() -> None:
    long_reply = (
        "불멍은 정말 힐링이죠. 복잡한 일상에서 벗어나 온전히 나에게 집중하는 시간이 되니까요. "
        "반대로 캠핑 가서 이건 좀 아니다 싶은 건 있어요?"
    )
    assert len(long_reply) > CONCISE_REPLY_CHARS

    assert (
        sanitize_response(long_reply, ConversationGoal.DISLIKE, concise=True)
        == "반대로 캠핑 가서 이건 좀 아니다 싶은 건 있어요?"
    )
    assert sanitize_response(long_reply, ConversationGoal.DISLIKE) == long_reply


def test_reflective_prompts_carry_topic_motive_and_clues() -> None:
    state = reflective_state()
    state.thread_topic = "캠핑"
    state.profile.signals.append(
        signal(
            TasteField.PREFERENCES,
            "아무 생각 안 하는 시간",
            target="캠핑",
            path=("취향", "동기", "휴식·재충전"),
            aspect=PreferenceAspect.MOTIVE,
        )
    )

    system = build_reply_messages(state, ConversationGoal.DISLIKE, "좋아요")[0]["content"]
    contrast = build_reply_messages(state, ConversationGoal.DISLIKE, "좋아요")[-2]["content"]
    bridge = build_reply_messages(state, ConversationGoal.BRIDGE, "좋아요")[-2]["content"]
    reflect = build_reply_messages(state, ConversationGoal.REFLECT, "좋아요")[-2]["content"]

    assert "곰곰이 돌아보게" in system
    assert "지금 이야기 줄기: 캠핑" in contrast
    assert "아무 생각 안 하는 시간" in bridge
    assert "지금까지 들은 단서: 아무 생각 안 하는 시간" in reflect
    assert "곰곰이" in build_opening_messages(style=ConversationStyle.REFLECTIVE)[0]["content"]
    assert "곰곰이" not in build_opening_messages()[0]["content"]


def test_depth_questions_are_asked_only_in_reflective_conversation() -> None:
    reflective = reflective_state()
    reflective.history.append(
        ConversationTurn(role="assistant", content="주말엔 뭐 해요?", created_at=NOW)
    )
    explore = ConversationState(user_id=1, conversation_room_id=1)
    explore.history.append(
        ConversationTurn(role="assistant", content="주말엔 뭐 해요?", created_at=NOW)
    )

    assert "answer_depth" in build_judgment_request(reflective, "캠핑 가요", []).questions
    assert "answer_depth" not in build_judgment_request(explore, "캠핑 가요", []).questions

    delta, _ = judgment_delta([], {"answer_depth": {"type": "score", "score": 1.8}}, ("캠핑 가요",))
    assert delta.answer_depth == 1.8


def test_motive_is_stored_as_motive_taste() -> None:
    delta, _ = judgment_delta(
        [Candidate("아무 생각 안 하는 시간", CandidateKind.ATTRIBUTE, target="캠핑")],
        {
            "stance_0": {"type": "choice", "choice": "like", "confidence": 0.9},
            "material_0": {"type": "choice", "choice": "taste", "confidence": 0.8},
            "taste_axis_0": {"type": "choice", "choice": "motive.recharge", "confidence": 0.9},
        },
        ("불 피워 놓고 아무 생각 안 할 때가 좋아요",),
    )

    (item,) = delta.items
    assert item.field == TasteField.PREFERENCES
    assert item.aspect == PreferenceAspect.MOTIVE
    assert item.taxonomy_path == ("취향", "동기", "휴식·재충전")


def test_reflective_state_round_trips_through_storage() -> None:
    state = reflective_state()
    state.thread_topic = "캠핑"
    state.thread_stage = ThreadStage.WHY
    state.thread_turns = 2
    state.why_count = 1
    state.last_answer_depth = 1.5
    state.shallow_streak = 1
    state.reflected = True
    state.reflection_values = ["혼자 가는 캠핑"]

    restored = ConversationState.from_dict(state.to_dict())

    assert restored.conversation_style == ConversationStyle.REFLECTIVE
    assert (restored.thread_topic, restored.thread_stage, restored.thread_turns) == (
        "캠핑",
        ThreadStage.WHY,
        2,
    )
    assert (restored.why_count, restored.last_answer_depth, restored.shallow_streak) == (1, 1.5, 1)
    assert restored.reflected is True
    assert restored.reflection_values == ["혼자 가는 캠핑"]


def test_frequency_variants_of_the_same_interest_are_dropped() -> None:
    from app.application.taste_judgment import without_frequency_variants

    kept = without_frequency_variants(
        [
            Candidate("캠핑", CandidateKind.ACTIVITY),
            Candidate("주말 캠핑", CandidateKind.ACTIVITY),
            Candidate("매일 하는 뜨개질", CandidateKind.ACTIVITY),
            Candidate("혼자 가는 캠핑", CandidateKind.ATTRIBUTE, target="캠핑"),
        ]
    )

    # "매일 하는 뜨개질"은 뜨개질 후보가 따로 없어서 남긴다. 취향 후보는 건드리지 않는다.
    assert [candidate.subject for candidate in kept] == [
        "캠핑",
        "매일 하는 뜨개질",
        "혼자 가는 캠핑",
    ]


def test_frequency_variant_of_a_stored_interest_is_dropped() -> None:
    from app.application.taste_judgment import without_frequency_variants

    kept = without_frequency_variants(
        [Candidate("퇴근 후 핸드드립 커피", CandidateKind.ACTIVITY)],
        known={"핸드드립 커피"},
    )

    assert kept == []


def test_wrap_reply_ending_with_question_is_replaced_with_closing() -> None:
    from app.domain.conversation.guards import CLOSING_MESSAGE

    service = ConversationService(
        model_gateway=FakeGateway([{"items": []}]),
        extraction_model="extractor",
    )
    state = reflective_state()

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="네 맞아요",
            raw_reply="고마워요. 다음엔 어떤 캠핑장을 가 보고 싶어요?",
            goal=ConversationGoal.WRAP,
            completion_reason=CompletionReason.SUFFICIENT,
        )
    )

    assert completed.reply == f"고마워요. {CLOSING_MESSAGE}"
    assert completed.state.history[-1].content == completed.reply


def test_reflect_direction_is_not_overridden_by_user_story() -> None:
    state = reflective_state()
    _ready_profile(state)

    direction = build_reply_messages(state, ConversationGoal.REFLECT, "즉흥적으로 가요")[-2][
        "content"
    ]

    assert "반드시 이 방향대로" in direction
    assert "새 이야기를 꺼냈으면" not in direction
