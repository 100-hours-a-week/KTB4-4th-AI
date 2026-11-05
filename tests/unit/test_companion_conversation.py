import asyncio
from datetime import UTC, datetime

import pytest

from app.application.conversation_prompts import (
    COMPANION_MOVES,
    CORRECTION_NOTE,
    NEW_TOPIC,
    SYSTEM_COMPANION,
    build_opening_messages,
    build_reply_messages,
)
from app.application.conversation_service import ConversationService
from app.domain.conversation import policy
from app.domain.conversation.energy import (
    LOW_ENERGY,
    asks_back,
    corrects,
    turn_energy,
    wants_to_exit,
)
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
    assert turn_energy(state, "네") >= LOW_ENERGY


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


def test_unsure_answer_lowers_the_burden_without_guessing_preferences() -> None:
    state = companion_state(turn_count=2)
    card = build_reply_messages(
        state, ConversationGoal.CHAT, "좋아하는 거요? 딱히 없는데", move=ConversationMove.FOLLOW
    )[-1]["content"]

    assert "방금 모른다고 한 내용이나 이유를 다시 묻지 않는다" in card
    assert "질문은 하나만 한다" in card
    assert "혹시 ~ 좋아해요?" not in card


def test_puzzled_reply_is_not_treated_as_asking_back() -> None:
    assert not asks_back("어쩌라는거지?")
    assert not asks_back("뭔 소리예요?")
    assert asks_back("니쥬는 뭐 좋아해요?")


def test_user_asking_back_is_followed_and_answered() -> None:
    state = companion_state(turn_count=3)

    decision = decide_goal(state, "니쥬는 뭐 좋아해요?")
    assert decision.move == ConversationMove.FOLLOW
    card = build_reply_messages(state, decision.goal, "니쥬는 뭐 좋아해요?", move=decision.move)[
        -1
    ]["content"]
    assert "네 생각부터 솔직하게 답한다" in card


def test_low_energy_twice_lightens_with_choices() -> None:
    state = with_user_turns(companion_state(turn_count=4), "캠핑 진짜 좋아해요", "그냥요")
    state.user_length_avg = 20.0
    state.low_energy_streak = 1

    assert decide_goal(state, "음").move == ConversationMove.CHOICE


def test_short_answers_in_a_row_do_not_force_choices() -> None:
    state = with_user_turns(followed(companion_state(turn_count=2), "follow"), "치킨이요")

    decision = decide_goal(state, "없어요")
    assert decision.move == ConversationMove.FOLLOW
    assert decision.scene is None

    assert decide_goal(state, "난 매운 거 좋아해").move == ConversationMove.FOLLOW

    state.move_history = ["choice"]
    assert decide_goal(state, "짠 거요").move == ConversationMove.FOLLOW


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


def test_two_follow_ups_then_review_topic() -> None:
    state = followed(companion_state(turn_count=3), "choice", "follow")
    assert decide_goal(state, CALM_ANSWER).move == ConversationMove.FOLLOW

    state = followed(companion_state(turn_count=3), "choice", "follow", "follow")
    decision = decide_goal(state, CALM_ANSWER)
    assert decision.move == ConversationMove.SHIFT
    assert decision.scene is None


def test_excited_punctuation_does_not_disable_topic_review() -> None:
    state = followed(companion_state(turn_count=5), "choice", "follow")
    assert decide_goal(state, EXCITED_ANSWER).move == ConversationMove.FOLLOW

    state = followed(companion_state(turn_count=5), "choice", "follow", "follow")
    assert decide_goal(state, EXCITED_ANSWER).move == ConversationMove.SHIFT


def test_topic_review_does_not_force_a_choice_or_ignore_new_information() -> None:
    state = followed(companion_state(turn_count=3), "follow", "follow")
    decision = decide_goal(state, CALM_ANSWER)
    assert decision.move == ConversationMove.SHIFT
    card = build_reply_messages(state, decision.goal, CALM_ANSWER, move=decision.move)[-1][
        "content"
    ]
    assert NEW_TOPIC in card
    assert "최신 사용자 말이 우선" in card
    assert "완전히 다른 소재" not in card


def test_unsure_answer_gets_choices_instead_of_the_same_question() -> None:
    state = followed(companion_state(turn_count=2), "follow")

    decision = decide_goal(state, "음 잘 모르겠어요")
    assert decision.move == ConversationMove.CHOICE
    card = build_reply_messages(
        state, decision.goal, "음 잘 모르겠어요", move=decision.move, scene=decision.scene
    )[-1]["content"]
    assert card.startswith("사용자가 잘 모르겠다고 했다.")
    assert NEW_TOPIC not in card
    assert "현재 이야기의 쉬운 행동이나 장면" in card


def test_answer_to_choices_is_followed_like_any_other_answer() -> None:
    state = followed(companion_state(turn_count=4), "follow", "choice")

    decision = decide_goal(state, "바다요")
    assert decision.move == ConversationMove.FOLLOW
    card = build_reply_messages(state, decision.goal, "바다요", move=decision.move)[-1]["content"]
    assert "직전 질문과 이번 답을 함께 읽고" in card


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


def test_asking_back_takes_priority_over_topic_review() -> None:
    state = followed(companion_state(turn_count=6), *["follow"] * 3)

    decision = decide_goal(state, "니쥬는 캠핑 가봤어요?")
    assert decision.move == ConversationMove.FOLLOW
    card = build_reply_messages(
        state, decision.goal, "니쥬는 캠핑 가봤어요?", move=decision.move, scene=decision.scene
    )[-1]["content"]
    assert "네 생각부터 솔직하게 답한다" in card
    assert NEW_TOPIC not in card


def test_corrects_detects_user_fixing_nijyu() -> None:
    for answer in (
        "그건 또 아니야",
        "아니 겨울이 아직 안왔는데",
        "지난번에 호빵 겨울에 먹는다했잖아",
        "꼭 그런 건 아닌데",
        "캠핑 아니고 등산이요",
        "어쩌라는거지?",
    ):
        assert corrects(answer), answer
    for answer in (
        "아니요",
        "난 매운 거 좋아해",
        "바다 아니면 산이요",
        "중간",
        "그 캠핑장 진짜 장난 아니야",
        "맛뿐만 아니라 분위기도 좋아요",
    ):
        assert not corrects(answer), answer


def test_correction_stays_on_topic_even_when_follow_ups_are_used_up() -> None:
    state = followed(companion_state(turn_count=5), "choice", "follow", "follow")

    decision = decide_goal(state, "그건 또 아니야")
    assert decision.move == ConversationMove.FOLLOW
    assert decision.scene is None
    card = build_reply_messages(state, decision.goal, "그건 또 아니야", move=decision.move)[-1][
        "content"
    ]
    assert card == CORRECTION_NOTE


def test_unsure_correction_is_handled_before_choices() -> None:
    state = followed(companion_state(turn_count=2), "follow")

    decision = decide_goal(state, "그건 아닌데 잘 모르겠어요")
    assert decision.move == ConversationMove.FOLLOW
    card = build_reply_messages(
        state, decision.goal, "그건 아닌데 잘 모르겠어요", move=decision.move
    )[-1]["content"]
    assert card == CORRECTION_NOTE


def test_topic_notes_allow_relevant_memory_and_keep_choices_on_topic() -> None:
    assert NEW_TOPIC in COMPANION_MOVES[ConversationMove.SHIFT]
    assert NEW_TOPIC not in COMPANION_MOVES[ConversationMove.CHOICE]
    assert "메모가 회상을 요청할 때" in SYSTEM_COMPANION


def _topic_conversation() -> ConversationState:
    state = followed(companion_state(turn_count=4), "follow", "shift", "follow")
    state.history = [
        ConversationTurn(role=role, content=content, created_at=NOW)
        for role, content in (
            ("assistant", "요즘 뭐 먹어요?"),
            ("user", "치킨이요"),
            ("assistant", "어떤 맛이요?"),
            ("user", "양념이요"),
            ("assistant", "노래는 뭐 들어요?"),
            ("user", "재즈요"),
            ("assistant", "언제 들어요?"),
        )
    ]
    return state


@pytest.mark.parametrize(
    "move", [ConversationMove.FOLLOW, ConversationMove.SHIFT, ConversationMove.CHOICE]
)
def test_question_answer_context_is_kept_across_topic_moves(move: ConversationMove) -> None:
    state = _topic_conversation()
    messages = build_reply_messages(state, ConversationGoal.CHAT, "밤에요", move=move)
    actual = [(m["role"], m["content"]) for m in messages if m["role"] != "system"]
    assert actual == [(t.role, t.content) for t in state.history] + [("user", "밤에요")]


def test_prompt_has_no_fixed_topic_change_question() -> None:
    for phrase in ("자주 듣는 노래", "단 거랑 짠 거", "끌어와 엮지"):
        assert phrase not in SYSTEM_COMPANION
        assert not any(phrase in note for note in COMPANION_MOVES.values())


# ---- 종료 ----


def test_ready_profile_does_not_interrupt_new_user_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    ready(monkeypatch)
    state = companion_state(turn_count=5)
    state.energy = 0.8
    state.peak_energy = 0.8
    state.user_length_avg = 20.0

    decision = decide_goal(state, "요즘 러닝에 빠져서 매일 뛰어요 ㅋㅋ")
    assert decision.goal == ConversationGoal.CHAT


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
    card = messages[-1]["content"]
    assert COMPANION_MOVES[ConversationMove.ADD] in card
    assert not any('"items"' in message["content"] for message in messages)
    assert messages[-2] == {"role": "user", "content": "요즘 뜨개질 해요"}


def test_turn_note_does_not_split_question_and_answer() -> None:
    state = with_user_turns(companion_state(turn_count=2), "산이 좋아요")
    state.history.append(
        ConversationTurn(
            role="assistant", content="단 거랑 짠 거 중에 뭐가 좋아요?", created_at=NOW
        )
    )

    messages = build_reply_messages(
        state, ConversationGoal.CHAT, "중간", move=ConversationMove.FOLLOW
    )

    assert messages[-3]["role"] == "assistant"
    assert messages[-3]["content"] == "단 거랑 짠 거 중에 뭐가 좋아요?"
    assert messages[-2] == {"role": "user", "content": "중간"}
    assert messages[-1]["role"] == "system"


def test_reply_prompt_tells_today_in_korean_time() -> None:
    # 10월 31일 23시(UTC)는 한국 기준 11월 1일이다.
    now = datetime(2026, 10, 31, 23, tzinfo=UTC)

    messages = build_reply_messages(companion_state(), ConversationGoal.CHAT, "네", now=now)

    assert any("오늘은 2026년 11월 1일이다." in message["content"] for message in messages)


def test_opening_uses_companion_prompt() -> None:
    messages = build_opening_messages()

    assert messages[0]["content"] == SYSTEM_COMPANION
    assert "바로 답할 수 있게" in messages[1]["content"]


# ---- 응답 가드 ----


def test_companion_reply_ends_at_one_question() -> None:
    no_question = "불 끄고 혼자 보는 영화는 거의 의식이에요. 저는 잔잔한 쪽이 진리라고 봐요."
    assert sanitize_response(no_question, ConversationGoal.CHAT) == no_question

    play = "평생 영화관 vs 평생 빔프로젝터, 뭐 골라요? 저는 빔프요, 잠옷이 허락되니까."
    assert sanitize_response(play, ConversationGoal.CHAT) == play.split("?", 1)[0] + "?"

    explanation = (
        "어떻게 생각해요? 저는 직접 경험해 본 적은 없지만, "
        "일정을 비워 두면 그날 마음에 드는 곳에서 더 오래 머물 수 있을 것 같아요."
    )
    assert sanitize_response(explanation, ConversationGoal.CHAT) == "어떻게 생각해요?"


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
