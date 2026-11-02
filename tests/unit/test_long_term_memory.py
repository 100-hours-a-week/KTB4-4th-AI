from dataclasses import replace
from datetime import UTC, datetime, timedelta

from app.application.conversation_prompts import (
    RETURNING_OPENING_INSTRUCTION,
    build_reply_messages,
)
from app.application.conversation_service import ConversationService
from app.application.taste_judgment import correction_targets
from app.domain.conversation.models import ConversationGoal, ConversationMove, ConversationState
from app.domain.conversation.policy import (
    MIN_RECALL_TURN,
    conversation_progress,
    decide_goal,
    recommendation_readiness,
    record_companion_turn,
)
from app.domain.profile.memory import (
    FORGET_CONFIDENCE,
    HALF_LIFE_DAYS,
    LongTermMemory,
    combined_profile,
    dropped_memory_keys,
    effective_confidence,
    merge_memory,
    opening_topic,
    recheck_topic,
)
from app.domain.profile.merger import link_role_for, visibility_for
from app.domain.profile.models import (
    DropRef,
    EvidenceType,
    ProfileSignal,
    ProfileState,
    SignalStatus,
    TasteField,
)

NOW = datetime(2026, 11, 2, tzinfo=UTC)


def signal(
    field: TasteField,
    value: str,
    confidence: float = 0.9,
    *,
    days_ago: int = 0,
    mentions: int = 1,
) -> ProfileSignal:
    seen = NOW - timedelta(days=days_ago)
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=confidence,
        link_role=link_role_for(field),
        visibility=visibility_for(field),
        intent_type=None,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=seen,
        updated_at=seen,
        mention_count=mentions,
    )


def memory(*signals: ProfileSignal, recalled: list[str] | None = None) -> LongTermMemory:
    return LongTermMemory(
        user_id=1,
        signals=list(signals),
        session_count=1,
        updated_at=NOW - timedelta(days=7),
        recalled=recalled or [],
    )


def by_key(signals: list[ProfileSignal]) -> dict[tuple[TasteField, str], ProfileSignal]:
    return {(item.field, item.normalized_value): item for item in signals}


def test_confidence_halves_every_half_life_but_safety_items_never_fade() -> None:
    camping = signal(TasteField.HOBBIES, "캠핑", 0.8, days_ago=int(HALF_LIFE_DAYS))
    allergy = signal(TasteField.CONSTRAINTS, "견과류", 0.8, days_ago=400)

    assert round(effective_confidence(camping, NOW), 3) == 0.4
    assert effective_confidence(allergy, NOW) == 0.8


def test_merge_refreshes_reconfirmed_items_and_keeps_raw_confidence_for_the_rest() -> None:
    old = memory(
        signal(TasteField.HOBBIES, "캠핑", 0.8, days_ago=60, mentions=2),
        signal(TasteField.INTERESTS, "커피", 0.7, days_ago=30),
        # 1년 전에 들은 것은 잊는다.
        signal(TasteField.INTERESTS, "뜨개질", 0.9, days_ago=365),
    )
    session = ProfileState(signals=[signal(TasteField.HOBBIES, "캠핑", 0.75)])

    merged = merge_memory(old, session, user_id=1, now=NOW, recalled=["캠핑"])
    signals = by_key(merged.signals)

    camping = signals[(TasteField.HOBBIES, "캠핑")]
    assert camping.updated_at == NOW
    assert camping.mention_count == 3
    # 기억의 지금 확신도(0.8 → 약 0.5)보다 이번 대화가 높으므로 이번 값이다.
    assert camping.confidence == 0.75
    coffee = signals[(TasteField.INTERESTS, "커피")]
    # 다시 나오지 않은 기억은 원래 값과 시각을 그대로 둬서 감쇠가 겹치지 않는다.
    assert coffee.confidence == 0.7
    assert coffee.updated_at == NOW - timedelta(days=30)
    assert (TasteField.INTERESTS, "뜨개질") not in signals
    assert merged.session_count == 2
    assert merged.recalled == ["캠핑"]


def test_merge_lets_this_conversation_overrule_old_memory() -> None:
    old = memory(
        signal(TasteField.INTERESTS, "향수"),
        signal(TasteField.WANTS, "캠핑 의자"),
        signal(TasteField.INTERESTS, "등산"),
        signal(TasteField.INTERESTS, "볼링"),
    )
    dropped_hiking = signal(TasteField.INTERESTS, "등산")
    dropped_hiking.status = SignalStatus.SUPERSEDED
    session = ProfileState(
        signals=[
            signal(TasteField.DISLIKES, "향수"),
            signal(TasteField.OWNED, "캠핑 의자"),
            dropped_hiking,
        ]
    )

    merged = merge_memory(
        old,
        session,
        user_id=1,
        now=NOW,
        dropped=[(TasteField.INTERESTS, "볼링")],
    )
    keys = set(by_key(merged.signals))

    assert keys == {(TasteField.DISLIKES, "향수"), (TasteField.OWNED, "캠핑 의자")}


def test_first_conversation_becomes_the_first_memory() -> None:
    session = ProfileState(signals=[signal(TasteField.HOBBIES, "캠핑")], axes=["혼자"])

    merged = merge_memory(None, session, user_id=7, now=NOW)

    assert merged.user_id == 7
    assert merged.session_count == 1
    assert [item.value for item in merged.signals] == ["캠핑"]
    assert merged.axes == ["혼자"]


def test_combined_profile_is_the_session_profile_without_memory() -> None:
    session = ProfileState(signals=[signal(TasteField.HOBBIES, "캠핑")])

    assert combined_profile(None, session, now=NOW) is session


def test_combined_profile_carries_faded_confidence_for_analysis() -> None:
    old = memory(signal(TasteField.INTERESTS, "커피", 0.8, days_ago=int(HALF_LIFE_DAYS)))
    session = ProfileState(signals=[signal(TasteField.HOBBIES, "캠핑")])

    profile = combined_profile(old, session, now=NOW)

    assert {item.value: round(item.confidence, 2) for item in profile.stored_signals()} == {
        "커피": 0.4,
        "캠핑": 0.9,
    }


def test_opening_and_recheck_topics_split_by_confidence_and_skip_recent_ones() -> None:
    old = memory(
        signal(TasteField.HOBBIES, "캠핑", 0.9),
        signal(TasteField.INTERESTS, "커피", 0.85),
        signal(TasteField.INTERESTS, "보드게임", 0.9, days_ago=60),
        signal(TasteField.UNAFFORDABLE, "카메라", 0.9),
        recalled=["캠핑"],
    )

    assert opening_topic(old, NOW) == "커피"
    assert recheck_topic(old, NOW, exclude=[]) == "보드게임"
    assert recheck_topic(old, NOW, exclude=["보드게임"]) is None


def test_dropped_memory_keys_only_names_remembered_values() -> None:
    old = memory(signal(TasteField.HOBBIES, "캠핑"))

    keys = dropped_memory_keys(
        old,
        [DropRef(TasteField.HOBBIES, "캠핑"), DropRef(TasteField.HOBBIES, "낚시")],
    )

    assert keys == [(TasteField.HOBBIES, "캠핑")]


def test_memory_survives_session_serialization() -> None:
    state = ConversationState(
        user_id=1, conversation_room_id=2, memory=memory(signal(TasteField.HOBBIES, "캠핑"))
    )
    state.memory_dropped.append((TasteField.HOBBIES, "캠핑"))

    restored = ConversationState.from_dict(state.to_dict())

    assert restored.memory is not None
    assert restored.memory.signals[0].value == "캠핑"
    assert restored.memory_dropped == [(TasteField.HOBBIES, "캠핑")]


def test_returning_user_opening_mentions_one_remembered_topic() -> None:
    service = ConversationService(model_gateway=None, extraction_model="test")  # type: ignore[arg-type]

    prepared = service.prepare_session(
        user_id=1,
        conversation_room_id=2,
        memory=memory(signal(TasteField.HOBBIES, "캠핑")),
        now=NOW,
    )

    instruction = prepared.messages[1]["content"]
    assert instruction.startswith(RETURNING_OPENING_INSTRUCTION.split("{absence}")[0])
    assert "'캠핑'" in instruction
    assert prepared.state.memory_recalled == ["캠핑"]


def test_first_time_user_gets_the_usual_opening() -> None:
    service = ConversationService(model_gateway=None, extraction_model="test")  # type: ignore[arg-type]

    prepared = service.prepare_session(user_id=1, conversation_room_id=2, now=NOW)

    assert "다시 찾아온" not in prepared.messages[1]["content"]
    assert prepared.state.memory_recalled == []


def test_memory_does_not_count_toward_progress_or_readiness() -> None:
    state = ConversationState(
        user_id=1,
        conversation_room_id=2,
        memory=memory(*(signal(TasteField.INTERESTS, f"관심{index}") for index in range(8))),
    )

    assert conversation_progress(state) == 0
    assert not recommendation_readiness(state).sufficient


def _state_ready_to_shift(old: LongTermMemory) -> ConversationState:
    state = ConversationState(
        user_id=1,
        conversation_room_id=2,
        memory=old,
        turn_count=MIN_RECALL_TURN,
        last_active_at=NOW,
        move_history=["follow", "follow", "follow"],
    )
    state.energy = 0.2
    return state


def test_fading_memory_is_rechecked_once_when_the_topic_changes() -> None:
    state = _state_ready_to_shift(
        memory(signal(TasteField.INTERESTS, "보드게임", 0.9, days_ago=60))
    )

    decision = decide_goal(state, "")
    assert decision.goal == ConversationGoal.CHAT
    assert decision.move == ConversationMove.RECALL
    assert decision.scene == "보드게임"

    record_companion_turn(
        state,
        utterance="네",
        move=decision.move,
        scene=decision.scene,
        answer_depth=None,
        new_interests=[],
    )
    state.move_history = ["follow", "follow", "follow"]
    assert state.memory_rechecked
    assert decide_goal(state, "").move != ConversationMove.RECALL


def test_recheck_waits_until_this_conversation_has_its_own_story() -> None:
    state = _state_ready_to_shift(
        memory(signal(TasteField.INTERESTS, "보드게임", 0.9, days_ago=60))
    )
    state.turn_count = MIN_RECALL_TURN - 1

    assert decide_goal(state, "").move != ConversationMove.RECALL


def test_reply_prompt_lists_known_memory_not_heard_in_this_conversation() -> None:
    state = ConversationState(
        user_id=1,
        conversation_room_id=2,
        memory=memory(
            signal(TasteField.HOBBIES, "캠핑"),
            signal(TasteField.INTERESTS, "커피"),
            signal(TasteField.INTERESTS, "보드게임", 0.9, days_ago=60),
        ),
        last_active_at=NOW,
        profile=ProfileState(signals=[signal(TasteField.HOBBIES, "캠핑")]),
    )

    messages = build_reply_messages(state, ConversationGoal.CHAT, "요즘 바빠요")
    memo = next(m["content"] for m in messages if m["content"].startswith("지난 대화에서"))

    assert "커피(관심)" in memo
    assert "캠핑" not in memo
    # 흐려진 기억은 이미 아는 것으로 보지 않는다. 다시 확인하기 후보다.
    assert "보드게임" not in memo


def test_mentioned_memory_becomes_a_correction_target() -> None:
    state = ConversationState(
        user_id=1,
        conversation_room_id=2,
        memory=memory(signal(TasteField.HOBBIES, "캠핑"), signal(TasteField.INTERESTS, "커피")),
        last_active_at=NOW,
    )

    targets = correction_targets(state, "요즘은 캠핑 안 가요")

    assert [target.value for target in targets] == ["캠핑"]


def test_forgotten_memory_is_not_a_correction_target() -> None:
    faded = replace(
        signal(TasteField.HOBBIES, "캠핑", FORGET_CONFIDENCE),
        updated_at=NOW - timedelta(days=30),
    )
    state = ConversationState(
        user_id=1, conversation_room_id=2, memory=memory(faded), last_active_at=NOW
    )

    assert correction_targets(state, "요즘은 캠핑 안 가요") == []
