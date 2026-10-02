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
from app.domain.conversation.energy import LOW_ENERGY, turn_energy, wants_to_exit
from app.domain.conversation.guards import sanitize_response
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationMove,
    ConversationState,
    ConversationStyle,
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
    recommendation_readiness,
    record_companion_turn,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def companion_state(*, turn_count: int = 0) -> ConversationState:
    return ConversationState(
        user_id=1,
        conversation_room_id=1,
        conversation_style=ConversationStyle.COMPANION,
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


def test_two_questions_in_a_row_switch_to_add() -> None:
    state = companion_state(turn_count=3)
    state.question_streak = 2

    assert decide_goal(state, "영화 보는 거 좋아해요").move == ConversationMove.ADD


def test_user_asking_back_is_followed_and_answered() -> None:
    state = companion_state(turn_count=3)
    state.question_streak = 2

    decision = decide_goal(state, "니쥬는 뭐 좋아해요?")
    assert decision.move == ConversationMove.FOLLOW
    card = build_reply_messages(state, decision.goal, "니쥬는 뭐 좋아해요?", move=decision.move)[
        -2
    ]["content"]
    assert "니쥬 생각부터" in card


def test_low_energy_twice_lightens_with_play_then_return() -> None:
    state = with_user_turns(companion_state(turn_count=4), "캠핑 진짜 좋아해요", "그냥요")
    state.user_length_avg = 20.0
    state.low_energy_streak = 1

    play = decide_goal(state, "음")
    assert play.move == ConversationMove.PLAY
    assert play.scene in {scene for scene, _ in COMPANION_SCENES}

    state.best_topic = "야구"
    back = decide_goal(state, "음")
    assert back.move == ConversationMove.RETURN
    assert back.scene == "야구"


def test_stays_on_topic_while_energy_is_high() -> None:
    state = companion_state(turn_count=6)
    state.turns_since_new_interest = 5
    state.energy = 0.9
    state.peak_energy = 0.9
    state.user_length_avg = 20.0

    decision = decide_goal(state, "그 캠핑장 별이 진짜 장난 아니었어요 ㅋㅋ 또 가고 싶어요")
    assert decision.move == ConversationMove.FOLLOW


def test_bridges_when_topic_runs_long_and_energy_falls() -> None:
    state = companion_state(turn_count=6)
    state.turns_since_new_interest = 3
    state.energy = 0.5
    state.peak_energy = 0.9
    state.user_length_avg = 20.0

    decision = decide_goal(state, "그냥 그랬어요 뭐 별로 특별한 건 없었고")
    assert decision.move == ConversationMove.BRIDGE
    assert decision.scene is not None


def test_late_turns_prefer_scenes_that_fill_gaps() -> None:
    state = companion_state(turn_count=15)
    missing = ("gear",)

    scene = policy._pick_scene(state, missing)
    gaps = dict(COMPANION_SCENES)[scene]
    assert "gear" in gaps


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


def test_companion_readiness_does_not_require_dislikes() -> None:
    state = companion_state()
    assert "exclusion" not in recommendation_readiness(state).missing_signals

    state.conversation_style = ConversationStyle.EXPLORE
    assert "exclusion" in recommendation_readiness(state).missing_signals


def test_progress_never_goes_down() -> None:
    state = companion_state()
    state.progress_floor = 40

    assert conversation_progress(state) == 40


# ---- 턴 기록 ----


def test_record_turn_tracks_questions_moves_and_scenes() -> None:
    state = companion_state()

    record_companion_turn(
        state,
        utterance="주말에 혼자 캠핑 갔어요",
        reply="혼캠이라니 멋있어요. 제일 좋았던 밤은요?",
        move=ConversationMove.FOLLOW,
        scene=None,
        answer_depth=None,
        new_interests=["캠핑"],
    )
    assert state.question_streak == 1
    assert state.best_topic == "캠핑"
    assert state.user_length_avg == len("주말에 혼자 캠핑 갔어요")

    record_companion_turn(
        state,
        utterance="음",
        reply="갑자기 밸런스 게임이요. 산 vs 바다? 저는 바다요.",
        move=ConversationMove.PLAY,
        scene="이번 주말에 하고 싶은 것",
        answer_depth=None,
        new_interests=[],
    )
    record_companion_turn(
        state,
        utterance="바다요",
        reply="바다파 반가워요. 파도 소리는 그냥 무료 명상이에요.",
        move=ConversationMove.ADD,
        scene=None,
        answer_depth=None,
        new_interests=[],
    )
    assert state.question_streak == 0
    assert state.move_history == ["follow", "play", "add"]
    assert state.used_scenes == ["이번 주말에 하고 싶은 것"]


# ---- 프롬프트 ----


def test_move_card_and_memo_replace_state_json() -> None:
    state = companion_state(turn_count=2)
    messages = build_reply_messages(
        state, ConversationGoal.CHAT, "요즘 뜨개질 해요", move=ConversationMove.ADD
    )

    assert messages[0]["content"] == SYSTEM_COMPANION
    card = messages[-2]["content"]
    assert COMPANION_MOVES[ConversationMove.ADD] in card
    assert "60자 안팎" in card
    assert not any('"items"' in message["content"] for message in messages)
    assert messages[-1] == {"role": "user", "content": "요즘 뜨개질 해요"}


def test_reply_budget_grows_with_user_length() -> None:
    state = companion_state(turn_count=2)
    long_message = (
        "주말마다 혼자 캠핑을 가는데 이번엔 강원도 쪽 계곡 옆에 자리를 잡아서 "
        "밤새 물소리 들으면서 잤어요. 아침에 커피 내려 마시는데 진짜 행복했어요"
    )
    card = build_reply_messages(
        state, ConversationGoal.CHAT, long_message, move=ConversationMove.FOLLOW
    )[-2]["content"]

    assert "110자 안팎" in card


def test_bridge_card_carries_scene() -> None:
    state = companion_state(turn_count=5)
    card = build_reply_messages(
        state,
        ConversationGoal.CHAT,
        "그렇죠",
        move=ConversationMove.BRIDGE,
        scene="요즘 장바구니에 담아 둔 것",
    )[-2]["content"]

    assert "요즘 장바구니에 담아 둔 것" in card


def test_opening_uses_companion_prompt() -> None:
    messages = build_opening_messages(now=NOW, style=ConversationStyle.COMPANION)

    assert messages[0]["content"] == SYSTEM_COMPANION
    assert "3초" in messages[1]["content"]


# ---- 응답 가드 ----


def test_companion_reply_keeps_statement_and_short_tail_after_question() -> None:
    no_question = "불 끄고 혼자 보는 영화는 거의 의식이에요. 저는 잔잔한 쪽이 진리라고 봐요."
    assert sanitize_response(no_question, ConversationGoal.CHAT, companion=True) == no_question

    play = "평생 영화관 vs 평생 빔프로젝터, 뭐 골라요? 저는 빔프요, 잠옷이 허락되니까."
    assert sanitize_response(play, ConversationGoal.CHAT, companion=True) == play

    rambling = (
        "제일 좋았던 밤은요? 저는 그런 밤이면 별 보면서 노래 듣고 "
        "핫초코 마시고 담요 덮고 있고 싶어요."
    )
    assert (
        sanitize_response(rambling, ConversationGoal.CHAT, companion=True) == "제일 좋았던 밤은요?"
    )


# ---- 서비스 ----


def test_service_records_move_and_energy() -> None:
    class Gateway:
        async def structured(self, *args, **kwargs):
            return {"items": []}

    service = ConversationService(
        model_gateway=Gateway(),  # type: ignore[arg-type]
        extraction_model="extraction-model",
        conversation_style=ConversationStyle.COMPANION,
    )
    prepared = service.prepare_session(user_id=1, conversation_room_id=1, now=NOW)
    state = prepared.state
    assert state.conversation_style == ConversationStyle.COMPANION

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
    assert restored.conversation_style == ConversationStyle.COMPANION
    assert restored.last_move == ConversationMove.ADD
    assert restored.energy == 0.7
    assert restored.best_topic == "캠핑"
    assert restored.used_scenes == ["이번 주말에 하고 싶은 것"]
    assert restored.progress_floor == 30
