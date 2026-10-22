import asyncio
from datetime import UTC, datetime

import pytest

from app.application.conversation_prompts import (
    COMPANION_MOVES,
    SYSTEM_COMPANION,
    build_opening_messages,
    build_reply_messages,
)
from app.application.conversation_service import ConversationService
from app.domain.conversation import policy
from app.domain.conversation.energy import LOW_ENERGY, asks_back, turn_energy, wants_to_exit
from app.domain.conversation.guards import sanitize_response
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationMove,
    ConversationState,
    ConversationTurn,
    ReadinessResult,
    SessionStatus,
)
from app.domain.conversation.policy import (
    COMPANION_CLOSE_AFTER_TURN,
    COMPANION_SCENES,
    MAX_TURNS,
    conversation_progress,
    decide_goal,
    record_companion_turn,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def companion_state(*, turn_count: int = 0) -> ConversationState:
    return ConversationState(
        user_id=1,
        conversation_room_id=1,
        turn_count=turn_count,
    )


def with_user_turns(state: ConversationState, *messages: str) -> ConversationState:
    for message in messages:
        state.history.append(ConversationTurn(role="user", content=message, created_at=NOW))
        state.history.append(ConversationTurn(role="assistant", content="응답", created_at=NOW))
    return state


def ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "recommendation_readiness", lambda state: ReadinessResult(True, ()))


# ---- 에너지 ----


def test_short_avoiding_answer_has_low_energy() -> None:
    state = companion_state()
    state.user_length_avg = 25.0

    assert turn_energy(state, "딱히요") < LOW_ENERGY
    assert turn_energy(state, "네") < LOW_ENERGY


def test_long_excited_answer_has_high_energy() -> None:
    state = companion_state()
    state.user_length_avg = 15.0

    assert turn_energy(state, "주말에 혼자 캠핑 갔는데 별이 진짜 쏟아지더라고요 ㅋㅋ") > 0.8


def test_yes_followed_by_story_is_not_avoidance() -> None:
    state = companion_state()
    state.user_length_avg = 15.0

    assert turn_energy(state, "네 맞아요 저 캠핑 진짜 좋아해요") > 0.6


def test_exit_intent() -> None:
    assert wants_to_exit("오늘은 이제 그만할게요")
    assert wants_to_exit("다음에 얘기해요")
    assert not wants_to_exit("그만큼 좋았어요")


# ---- move 선택 ----


def test_default_move_is_follow() -> None:
    state = companion_state(turn_count=1)

    decision = decide_goal(state, "주말에 캠핑 다녀왔어요. 날씨가 완전 좋았어요")
    assert decision.goal == ConversationGoal.CHAT
    assert decision.move == ConversationMove.FOLLOW


def test_unsure_answer_asks_with_a_suggestion() -> None:
    state = companion_state(turn_count=2)
    card = build_reply_messages(
        state, ConversationGoal.CHAT, "좋아하는 거요? 딱히 없는데", move=ConversationMove.FOLLOW
    )[-2]["content"]

    assert "혹시 ~ 좋아해요?" in card


def test_puzzled_reply_is_not_treated_as_asking_back() -> None:
    assert not asks_back("어쩌라는거지?")
    assert not asks_back("뭔 소리예요?")
    assert asks_back("니쥬는 뭐 좋아해요?")


def test_user_asking_back_is_followed_and_answered() -> None:
    state = companion_state(turn_count=3)

    decision = decide_goal(state, "니쥬는 뭐 좋아해요?")
    assert decision.move == ConversationMove.FOLLOW
    card = build_reply_messages(state, decision.goal, "니쥬는 뭐 좋아해요?", move=decision.move)[
        -2
    ]["content"]
    assert "네 생각부터 솔직하게 답한다" in card


def test_low_energy_twice_lightens_with_choices_then_return() -> None:
    state = with_user_turns(companion_state(turn_count=4), "캠핑 진짜 좋아해요", "그냥요")
    state.user_length_avg = 20.0
    state.low_energy_streak = 1

    choice = decide_goal(state, "음")
    assert choice.move == ConversationMove.CHOICE
    assert choice.scene in COMPANION_SCENES

    state.best_topic = "야구"
    back = decide_goal(state, "음")
    assert back.move == ConversationMove.RETURN
    assert back.scene == "야구"


def test_stays_on_topic_while_energy_is_high() -> None:
    state = companion_state(turn_count=6)
    state.energy = 0.9
    state.peak_energy = 0.9
    state.user_length_avg = 20.0

    decision = decide_goal(state, "그 캠핑장 별이 진짜 장난 아니었어요 ㅋㅋ 또 가고 싶어요")
    assert decision.move == ConversationMove.FOLLOW


def test_no_new_interest_does_not_force_a_topic_change() -> None:
    state = companion_state(turn_count=6)
    state.energy = 0.5
    state.peak_energy = 0.9
    state.user_length_avg = 20.0

    decision = decide_goal(
        state, "언젠가 책방을 열면 사람들이 편히 이야기할 수 있는 곳이면 좋겠어요"
    )
    assert decision.move == ConversationMove.FOLLOW
    assert decision.scene is None


def followed(state: ConversationState, *moves: str) -> ConversationState:
    state.user_length_avg = 20.0
    state.move_history = list(moves)
    return state


CALM_ANSWER = "네 친구랑 같이 갔어요"
EXCITED_ANSWER = "그 캠핑장 별이 진짜 장난 아니었어요 ㅋㅋ 또 가고 싶어요"


def test_two_follow_ups_then_shift_topic() -> None:
    state = followed(companion_state(turn_count=3), "follow")
    assert decide_goal(state, CALM_ANSWER).move == ConversationMove.FOLLOW

    state = followed(companion_state(turn_count=3), "choice", "follow", "follow")
    decision = decide_goal(state, CALM_ANSWER)
    assert decision.move == ConversationMove.SHIFT
    assert decision.scene in COMPANION_SCENES


def test_engaged_user_keeps_topic_up_to_four_follow_ups() -> None:
    for count in (2, 3):
        state = followed(companion_state(turn_count=5), *["follow"] * count)
        assert decide_goal(state, EXCITED_ANSWER).move == ConversationMove.FOLLOW

    state = followed(companion_state(turn_count=5), "choice", *["follow"] * 4)
    assert decide_goal(state, EXCITED_ANSWER).move == ConversationMove.SHIFT


def test_engaged_streak_ends_when_reaction_cools() -> None:
    state = followed(companion_state(turn_count=5), "choice", "follow", "follow", "follow")

    assert decide_goal(state, CALM_ANSWER).move == ConversationMove.SHIFT


def test_topic_done_offers_choices_when_none_were_offered_recently() -> None:
    state = followed(companion_state(turn_count=3), "follow", "follow")

    decision = decide_goal(state, CALM_ANSWER)
    assert decision.move == ConversationMove.CHOICE
    assert decision.scene in COMPANION_SCENES
    card = build_reply_messages(
        state, decision.goal, CALM_ANSWER, move=decision.move, scene=decision.scene
    )[-2]["content"]
    assert "열린 질문 대신 선택지를 준다" in card
    assert decision.scene in card


def test_unsure_answer_gets_choices_instead_of_the_same_question() -> None:
    state = followed(companion_state(turn_count=2), "follow")

    decision = decide_goal(state, "음 잘 모르겠어요")
    assert decision.move == ConversationMove.CHOICE
    card = build_reply_messages(
        state, decision.goal, "음 잘 모르겠어요", move=decision.move, scene=decision.scene
    )[-2]["content"]
    assert card.startswith("사용자가 잘 모르겠다고 했다.")
    assert "앞에서 사용자가 한 이야기" in card


def test_answer_to_choices_is_followed_like_any_other_answer() -> None:
    state = followed(companion_state(turn_count=4), "follow", "choice")

    decision = decide_goal(state, "바다요")
    assert decision.move == ConversationMove.FOLLOW
    card = build_reply_messages(state, decision.goal, "바다요", move=decision.move)[-2]["content"]
    assert card == COMPANION_MOVES[ConversationMove.FOLLOW]


def test_short_pick_from_choices_is_not_low_energy() -> None:
    state = followed(companion_state(turn_count=4), "choice")

    assert turn_energy(state, "바다") >= LOW_ENERGY
    record_companion_turn(
        state,
        utterance="바다",
        move=ConversationMove.FOLLOW,
        scene=None,
        answer_depth=None,
        new_interests=[],
    )
    assert state.low_energy_streak == 0
    assert state.user_length_avg == 20.0


def test_sessions_saved_with_the_old_move_name_still_load() -> None:
    state = followed(companion_state(turn_count=4), "follow", "play")

    assert state.last_move == ConversationMove.CHOICE


def test_topic_change_resets_follow_count_and_add_does_not() -> None:
    assert policy.topic_follow_count(followed(companion_state(), "follow", "shift")) == 0
    assert policy.topic_follow_count(followed(companion_state(), "choice", "follow")) == 1
    assert policy.topic_follow_count(followed(companion_state(), "follow", "add", "follow")) == 2


def test_asking_back_after_topic_is_done_answers_then_shifts() -> None:
    state = followed(companion_state(turn_count=6), *["follow"] * 4)

    decision = decide_goal(state, "니쥬는 캠핑 가봤어요?")
    assert decision.move == ConversationMove.SHIFT
    card = build_reply_messages(
        state, decision.goal, "니쥬는 캠핑 가봤어요?", move=decision.move, scene=decision.scene
    )[-2]["content"]
    assert "네 생각부터 솔직하게 답한다" in card
    assert "이 이야기는 더 묻지 않는다" in card


def test_shift_card_without_scene_uses_fallback() -> None:
    card = build_reply_messages(
        companion_state(turn_count=8),
        ConversationGoal.CHAT,
        CALM_ANSWER,
        move=ConversationMove.SHIFT,
    )[-2]["content"]

    assert "아직 이야기하지 않은 다른 일상" in card
    assert "방금 이야기" not in card


def test_conversation_scene_does_not_depend_on_missing_recommendation_fields(monkeypatch) -> None:
    scenes = []
    for missing in (("gear",), ("query_signals", "taste_axes")):
        monkeypatch.setattr(
            policy,
            "recommendation_readiness",
            lambda state, gaps=missing: ReadinessResult(False, gaps),
        )
        state = companion_state(turn_count=15)
        state.low_energy_streak = 1
        state.user_length_avg = 20
        scenes.append(decide_goal(state, "음").scene)

    assert scenes[0] is not None
    assert scenes[0] == scenes[1]


# ---- 종료 ----


def test_ready_but_still_fun_keeps_talking(monkeypatch: pytest.MonkeyPatch) -> None:
    ready(monkeypatch)
    state = companion_state(turn_count=5)
    state.energy = 0.8
    state.peak_energy = 0.8
    state.user_length_avg = 20.0

    assert decide_goal(state, "요즘 러닝에 빠져서 매일 뛰어요 ㅋㅋ").goal == ConversationGoal.CHAT


def test_ready_and_long_conversation_wraps(monkeypatch: pytest.MonkeyPatch) -> None:
    ready(monkeypatch)
    state = companion_state(turn_count=COMPANION_CLOSE_AFTER_TURN)

    decision = decide_goal(state, "좋아요")
    assert decision.goal == ConversationGoal.WRAP
    assert decision.completion_reason == CompletionReason.SUFFICIENT


def test_ready_and_energy_dropping_wraps(monkeypatch: pytest.MonkeyPatch) -> None:
    ready(monkeypatch)
    state = companion_state(turn_count=6)
    state.energy = 0.5
    state.peak_energy = 0.9

    assert decide_goal(state, "").goal == ConversationGoal.WRAP


def test_exit_intent_wraps_immediately() -> None:
    decision = decide_goal(companion_state(turn_count=2), "이제 그만할게요")
    assert decision.goal == ConversationGoal.WRAP
    assert decision.completion_reason == CompletionReason.USER_EXIT


def test_wraps_at_max_turns_and_after_lock() -> None:
    decision = decide_goal(companion_state(turn_count=MAX_TURNS), "")
    assert decision.completion_reason == CompletionReason.MAX_CYCLES

    state = companion_state()
    state.status = SessionStatus.INPUT_LOCKED
    assert decide_goal(state, "").goal == ConversationGoal.WRAP


def test_progress_never_goes_down() -> None:
    state = companion_state()
    state.progress_floor = 40

    assert conversation_progress(state) == 40


# ---- 턴 기록 ----


def test_record_turn_tracks_moves_and_scenes() -> None:
    state = companion_state()

    record_companion_turn(
        state,
        utterance="주말에 혼자 캠핑 갔어요",
        move=ConversationMove.FOLLOW,
        scene=None,
        answer_depth=None,
        new_interests=["캠핑"],
    )
    assert state.best_topic == "캠핑"
    assert state.user_length_avg == len("주말에 혼자 캠핑 갔어요")

    record_companion_turn(
        state,
        utterance="음",
        move=ConversationMove.CHOICE,
        scene="이번 주말에 하고 싶은 것",
        answer_depth=None,
        new_interests=[],
    )
    record_companion_turn(
        state,
        utterance="바다요",
        move=ConversationMove.ADD,
        scene=None,
        answer_depth=None,
        new_interests=[],
    )
    record_companion_turn(
        state,
        utterance="요즘은 그냥 집에 있어요",
        move=ConversationMove.SHIFT,
        scene="요즘 자주 먹는 것",
        answer_depth=None,
        new_interests=[],
    )
    assert state.move_history == ["follow", "choice", "add", "shift"]
    assert state.used_scenes == ["이번 주말에 하고 싶은 것", "요즘 자주 먹는 것"]


# ---- 프롬프트 ----


def test_move_card_and_memo_replace_state_json() -> None:
    state = companion_state(turn_count=2)
    messages = build_reply_messages(
        state, ConversationGoal.CHAT, "요즘 뜨개질 해요", move=ConversationMove.ADD
    )

    assert messages[0]["content"] == SYSTEM_COMPANION
    card = messages[-2]["content"]
    assert COMPANION_MOVES[ConversationMove.ADD] in card
    assert not any('"items"' in message["content"] for message in messages)
    assert messages[-1] == {"role": "user", "content": "요즘 뜨개질 해요"}


def test_choice_card_carries_scene() -> None:
    state = companion_state(turn_count=5)
    card = build_reply_messages(
        state,
        ConversationGoal.CHAT,
        "그렇죠",
        move=ConversationMove.CHOICE,
        scene="요즘 자주 먹는 것",
    )[-2]["content"]

    assert "요즘 자주 먹는 것" in card


def test_opening_uses_companion_prompt() -> None:
    messages = build_opening_messages()

    assert messages[0]["content"] == SYSTEM_COMPANION
    assert "바로 답할 수 있게" in messages[1]["content"]


# ---- 응답 가드 ----


def test_companion_reply_keeps_statement_and_short_tail_after_question() -> None:
    no_question = "불 끄고 혼자 보는 영화는 거의 의식이에요. 저는 잔잔한 쪽이 진리라고 봐요."
    assert sanitize_response(no_question, ConversationGoal.CHAT) == no_question

    play = "평생 영화관 vs 평생 빔프로젝터, 뭐 골라요? 저는 빔프요, 잠옷이 허락되니까."
    assert sanitize_response(play, ConversationGoal.CHAT) == play

    rambling = (
        "제일 좋았던 밤은요? 저는 그런 밤이면 별 보면서 노래 듣고 "
        "핫초코 마시고 담요 덮고 있고 싶어요."
    )
    assert sanitize_response(rambling, ConversationGoal.CHAT) == "제일 좋았던 밤은요?"


def test_companion_reply_drops_label_line_breaks_and_imitated_next_turn() -> None:
    raw = "니쥬: 와, 드디어 해방이네요.\n\n끝나자마자 뭐 했어요?\n사용자: 그냥 잤어요"
    assert (
        sanitize_response(raw, ConversationGoal.CHAT)
        == "와, 드디어 해방이네요. 끝나자마자 뭐 했어요?"
    )

    leaked = "그럴 때 있죠. 오늘은 쉬는 날이에요? [말투] 또래 친구와 메신저로"
    assert sanitize_response(leaked, ConversationGoal.CHAT) == "그럴 때 있죠. 오늘은 쉬는 날이에요?"


# ---- 서비스 ----


def test_service_records_move_and_energy() -> None:
    class Gateway:
        async def structured(self, *args, **kwargs):
            return {"items": []}

    service = ConversationService(
        model_gateway=Gateway(),  # type: ignore[arg-type]
        extraction_model="extraction-model",
    )
    prepared = service.prepare_session(user_id=1, conversation_room_id=1, now=NOW)
    state = prepared.state

    turn = service.prepare_turn(state, "주말에 혼자 캠핑 다녀왔어요 ㅋㅋ")
    assert turn.decision.move == ConversationMove.FOLLOW
    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="주말에 혼자 캠핑 다녀왔어요 ㅋㅋ",
            raw_reply="혼캠이라니 멋있어요. 제일 좋았던 순간은요?",
            goal=turn.decision.goal,
            move=turn.decision.move,
            now=NOW,
        )
    )

    assert completed.reply == "혼캠이라니 멋있어요. 제일 좋았던 순간은요?"
    assert completed.state.move_history == ["follow"]
    assert completed.state.energy is not None
    assert completed.state.status == SessionStatus.ACTIVE


def test_companion_state_round_trips_through_storage() -> None:
    state = companion_state(turn_count=3)
    state.move_history = ["follow", "add"]
    state.energy = 0.7
    state.best_topic = "캠핑"
    state.used_scenes = ["이번 주말에 하고 싶은 것"]
    state.progress_floor = 30

    restored = ConversationState.from_dict(state.to_dict())
    assert restored.last_move == ConversationMove.ADD
    assert restored.energy == 0.7
    assert restored.best_topic == "캠핑"
    assert restored.used_scenes == ["이번 주말에 하고 싶은 것"]
    assert restored.progress_floor == 30
