import json
from datetime import UTC, datetime

import pytest

from app.application.conversation_prompts import build_reply_messages
from app.domain.conversation.energy import LOW_ENERGY, asks_back, turn_energy, unsure, wants_to_exit
from app.domain.conversation.guards import sanitize_response, stable_reply_prefix
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationMove,
    ConversationState,
    ConversationTurn,
)
from app.domain.conversation.policy import decide_goal, record_companion_turn

NOW = datetime(2026, 11, 4, tzinfo=UTC)


@pytest.mark.parametrize(
    "utterance",
    [
        "야근을 그만하고 싶어요",
        "도자기는 나중에 할 생각이에요",
        "그 가게는 이제 됐고 다른 카페가 좋아요",
        "대화를 그만하고 싶지 않아요",
        "대화를 끝내지 말아 주세요",
        "친구가 그만하자고 했어요",
        '영화에서 "이제 그만할게요"라고 하더라고요',
        "그만큼 좋았어요",
    ],
)
def test_activity_plans_negation_and_quotes_do_not_close_chat(utterance: str) -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    assert not wants_to_exit(utterance)
    assert decide_goal(state, utterance).goal == ConversationGoal.CHAT


@pytest.mark.parametrize(
    "utterance",
    [
        "오늘은 이제 그만할게요",
        "이제 그만할래요",
        "대화를 그만하고 싶어요",
        "끝내주세요",
        "다음에 얘기해요",
        "나중에 다시 이야기해요",
        "이만 갈게요",
        "재미있었어요. 이제 그만할게요!",
    ],
)
def test_direct_exit_requests_still_close_chat(utterance: str) -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    assert wants_to_exit(utterance)
    assert decide_goal(state, utterance).goal == ConversationGoal.WRAP


@pytest.mark.parametrize(
    "utterance",
    [
        "너무 좋았는데 제목이 기억이 안 나요",
        "이름은 모르겠는데 바다가 보이는 캠핑장이었어요",
        "딱히 비싼 건 싫고 가벼운 게 좋아요",
    ],
)
def test_missing_details_do_not_discard_a_substantive_answer(utterance: str) -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    assert not unsure(utterance)
    assert decide_goal(state, utterance).move == ConversationMove.FOLLOW


def test_echoing_the_question_before_an_uncertain_answer_is_not_asking_back() -> None:
    utterance = "좋아하는 거요? 딱히 없는데"
    state = ConversationState(user_id=1, conversation_room_id=1)
    assert unsure(utterance)
    assert not asks_back(utterance)
    assert decide_goal(state, utterance).move == ConversationMove.CHOICE


def test_uncertainty_does_not_override_a_correction() -> None:
    utterance = "캠핑이 아니라 등산이고 산 이름은 기억이 안 나요"
    state = ConversationState(user_id=1, conversation_room_id=1, move_history=["follow"] * 2)
    decision = decide_goal(state, utterance)
    assert decision.move == ConversationMove.FOLLOW
    messages = build_reply_messages(state, decision.goal, utterance, move=decision.move)
    assert "정정한 내용은 유효하다" in messages[-1]["content"]


def test_repeated_specific_short_answers_are_not_treated_as_disengagement() -> None:
    state = ConversationState(user_id=1, conversation_room_id=1, user_length_avg=30)
    for utterance in ("혼자요", "한강이요", "밤이요", "네", "아니요"):
        decision = decide_goal(state, utterance)
        assert decision.goal == ConversationGoal.CHAT
        assert decision.move != ConversationMove.CHOICE
        assert turn_energy(state, utterance) >= LOW_ENERGY
        record_companion_turn(
            state,
            utterance=utterance,
            move=decision.move,
            scene=None,
            answer_depth=None,
            new_interests=[],
        )
        state.history.extend(
            [
                ConversationTurn("user", utterance, NOW),
                ConversationTurn("assistant", "다음 질문", NOW),
            ]
        )
        state.turn_count += 1
        assert state.low_energy_streak == 0


def test_uncertainty_after_choices_changes_the_question_without_repeating_it() -> None:
    state = ConversationState(user_id=1, conversation_room_id=1, user_length_avg=30)
    first = decide_goal(state, "잘 모르겠어요")
    assert first.move == ConversationMove.CHOICE
    record_companion_turn(
        state,
        utterance="잘 모르겠어요",
        move=first.move,
        scene=None,
        answer_depth=None,
        new_interests=[],
    )
    second = decide_goal(state, "글쎄요")
    assert second.move == ConversationMove.SHIFT
    messages = build_reply_messages(state, second.goal, "글쎄요", move=second.move)
    assert "현재 이야기와 이어지는 더 쉬운 행동이나 장면" in messages[-1]["content"]


@pytest.mark.parametrize(
    "utterance", ["말하기 싫어요", "그 얘기는 하고 싶지 않아요", "그냥 대답하기 싫어요"]
)
def test_declining_a_topic_changes_to_a_connected_question(utterance: str) -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    assert decide_goal(state, utterance).move == ConversationMove.SHIFT


@pytest.mark.parametrize("utterance", ["니쥬는 뭐 좋아해요?", "캠핑이 아니라 등산이요"])
def test_pending_completion_waits_for_direct_questions_and_corrections(utterance: str) -> None:
    state = ConversationState(
        user_id=1,
        conversation_room_id=1,
        pending_wrap=CompletionReason.SUFFICIENT,
    )
    assert decide_goal(state, utterance).move == ConversationMove.FOLLOW
    assert decide_goal(state, "네").goal == ConversationGoal.WRAP


def test_bounded_history_retains_question_answer_pairs_and_full_earlier_text() -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    for index in range(15):
        state.history.extend(
            [
                ConversationTurn("assistant", f"질문 {index}", NOW),
                ConversationTurn(
                    "user", f"답변 {index}: " + "긴 설명 " * 12 + "등산이 좋아요", NOW
                ),
            ]
        )
    state.history.append(ConversationTurn("assistant", "마지막 질문", NOW))
    messages = build_reply_messages(
        state, ConversationGoal.CHAT, "아까 말한 곳이요", move=ConversationMove.SHIFT
    )
    dialogue = [m for m in messages if m["role"] != "system"]
    assert dialogue[0]["role"] == "assistant"
    assert [(m["role"], m["content"]) for m in dialogue[:-1]] == [
        (t.role, t.content) for t in state.history[-19:]
    ]
    note = next(m["content"] for m in messages if m["content"].startswith("[이전 대화 참고 자료]"))
    earlier = json.loads(note.split("\n", 1)[1])
    assert [m["role"] for m in earlier] == ["assistant", "user"] * 4
    assert earlier[1]["content"] == state.history[5].content


def test_normal_bracketed_words_and_explanations_survive_streaming() -> None:
    raw = "[말투]가 편안한 사람이 좋다는 뜻이군요. 어떤 말투가 편해요?"
    assert sanitize_response(raw, ConversationGoal.CHAT) == raw
    emitted = ""
    for end in range(1, len(raw) + 1):
        stable = stable_reply_prefix(raw[:end], hold_question=False)
        if stable.startswith(emitted):
            emitted = stable
        assert raw.startswith(emitted)
    assert emitted == raw
