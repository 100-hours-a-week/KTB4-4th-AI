import asyncio

import pytest

from app.application.chat_use_cases import ChatUseCases, ReplyDelta, ResponseModelUnavailableError
from app.application.conversation_prompts import DISCOVERY_INTRO
from app.application.conversation_service import CompletedTurn, ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.taste_judgment import (
    Candidate,
    CandidateKind,
    build_judgment_request,
    judgment_delta,
)
from app.domain.conversation.models import ConversationState, ConversationTurn, SessionStatus
from app.domain.profile.merger import ProfileMerger
from app.domain.profile.models import EvidenceType, InsightKind, ProfileState, TasteField, utc_now
from app.infrastructure.persistence import InMemorySessionStore

PREVIOUS_QUESTION = "커피는 어떤 맛이 좋아요?"
CONNECTED_QUESTION = "그런 커피는 집에서 마셔요, 카페에서 마셔요?"


class Gateway:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.extraction_calls = 0
        self.chunks = []

    async def complete(self, messages, *, model):
        self.calls.append(list(messages))
        return self.replies.pop(0)

    async def structured(self, messages, **kwargs):
        self.extraction_calls += 1
        return {"items": []}

    async def stream(self, messages, *, model):
        for chunk in self.chunks:
            yield chunk


def setup(gateway):
    states = ConversationStateStore(InMemorySessionStore())
    use_cases = ChatUseCases(
        conversations=ConversationService(model_gateway=gateway, extraction_model="extractor"),
        states=states,
        model_gateway=gateway,
        response_model="response",
    )
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.history = [ConversationTurn("assistant", PREVIOUS_QUESTION, utc_now())]
    return use_cases, states, state


def test_opening_explains_hidden_taste_discovery_and_asks_one_question():
    async def run():
        gateway = Gateway("쉬는 날엔 주로 뭐 해요?")
        use_cases, _, _ = setup(gateway)
        started = await use_cases.create_session(user_id=1, conversation_room_id=101)
        assert started.greeting == f"{DISCOVERY_INTRO} 쉬는 날엔 주로 뭐 해요?"
        assert started.greeting.count("?") == 1

    asyncio.run(run())


def test_repeated_question_is_repaired_with_the_original_short_answer():
    async def run():
        gateway = Gateway(f"고소한 맛이 좋군요. {PREVIOUS_QUESTION}", CONNECTED_QUESTION)
        use_cases, states, state = setup(gateway)
        await states.save(state)
        completed = await use_cases.send_message(101, "고소한 거요", user_id=1)
        assert completed.reply == f"고소한 맛이 좋군요. {CONNECTED_QUESTION}"
        assert completed.reply.count("?") == 1
        assert len(gateway.calls) == 2
        assert gateway.extraction_calls == 1
        repair_context = gateway.calls[1]
        assert any(m["content"] == PREVIOUS_QUESTION for m in repair_context)
        assert any(m["content"] == "고소한 거요" for m in repair_context)
        assert "그 이유를 다시 묻지 않는다" in repair_context[-1]["content"]
        assert (await states.load(101)).turn_count == 1

    asyncio.run(run())


def test_failed_repair_keeps_the_user_turn_unsaved():
    async def run():
        gateway = Gateway(PREVIOUS_QUESTION, PREVIOUS_QUESTION)
        use_cases, states, state = setup(gateway)
        await states.save(state)
        with pytest.raises(ResponseModelUnavailableError, match="question_missing_or_repeated"):
            await use_cases.send_message(101, "고소한 거요", user_id=1)
        saved = await states.load(101)
        assert saved.turn_count == 0
        assert len(saved.history) == 1

    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("answer", "reply"),
    [
        ("그냥 좋은데요", "좋아하는 데 꼭 이유가 있어야 하는 건 아니니까요."),
        ("친구랑 같이 먹은 날이 좋았어요", "함께 먹었던 날이 좋은 기억으로 남았군요."),
        ("라테요", "거기서는 라테를 마시는군요."),
    ],
)
def test_reaction_without_question_stays_active_without_a_forced_follow_up(
    streaming, answer, reply
):
    async def run():
        gateway = Gateway(reply)
        gateway.chunks = [reply[:12], reply[12:]]
        use_cases, states, state = setup(gateway)
        await states.save(state)
        if streaming:
            events = [
                event
                async for event in use_cases.stream_message(
                    101, answer, user_id=1, user_message_id=5
                )
            ]
            completed = events[-1]
            assert isinstance(completed, CompletedTurn)
            assert "".join(e.text for e in events if isinstance(e, ReplyDelta)) == reply
            assert gateway.calls == []
        else:
            completed = await use_cases.send_message(101, answer, user_id=1)
            assert len(gateway.calls) == 1
        assert completed.reply == reply
        assert completed.state.status == SessionStatus.ACTIVE
        assert (await states.load(101)).turn_count == 1
        assert gateway.extraction_calls == 1

    asyncio.run(run())


def test_repair_can_stop_asking_when_the_subject_is_exhausted():
    async def run():
        gateway = Gateway(PREVIOUS_QUESTION, "그 맛이 마음에 드는군요.")
        use_cases, states, state = setup(gateway)
        await states.save(state)
        completed = await use_cases.send_message(101, "고소한 거요", user_id=1)
        assert completed.reply == "그 맛이 마음에 드는군요."
        assert len(gateway.calls) == 2
        assert completed.state.status == SessionStatus.ACTIVE

    asyncio.run(run())


def test_opening_still_repairs_a_missing_question():
    async def run():
        gateway = Gateway("반가워요.", "쉬는 날엔 주로 뭐 해요?")
        use_cases, _, _ = setup(gateway)
        started = await use_cases.create_session(user_id=1, conversation_room_id=101)
        assert started.greeting.count("?") == 1
        assert len(gateway.calls) == 2

    asyncio.run(run())


def test_multiple_questions_keep_only_the_first_without_an_extra_model_call():
    async def run():
        gateway = Gateway(f"고소한 맛이 좋군요. {CONNECTED_QUESTION} 어떤 원두예요?")
        use_cases, states, state = setup(gateway)
        await states.save(state)
        completed = await use_cases.send_message(101, "고소한 거요", user_id=1)
        assert completed.reply == f"고소한 맛이 좋군요. {CONNECTED_QUESTION}"
        assert len(gateway.calls) == 1

    asyncio.run(run())


def test_question_already_answered_before_the_previous_turn_is_also_repaired():
    async def run():
        gateway = Gateway(PREVIOUS_QUESTION, CONNECTED_QUESTION)
        use_cases, states, state = setup(gateway)
        state.history.extend(
            [
                ConversationTurn("user", "고소한 거요", utc_now()),
                ConversationTurn("assistant", "커피는 언제 마셔요?", utc_now()),
            ]
        )
        state.turn_count = 1
        await states.save(state)
        completed = await use_cases.send_message(101, "아침이요", user_id=1)
        assert completed.reply == CONNECTED_QUESTION
        assert len(gateway.calls) == 2

    asyncio.run(run())


def test_stream_never_emits_the_repeated_question_before_repair():
    async def run():
        gateway = Gateway(CONNECTED_QUESTION)
        gateway.chunks = ["고소한 맛이 좋군요. ", PREVIOUS_QUESTION]
        use_cases, states, state = setup(gateway)
        await states.save(state)
        events = [
            event
            async for event in use_cases.stream_message(
                101, "고소한 거요", user_id=1, user_message_id=5
            )
        ]
        text = "".join(event.text for event in events if isinstance(event, ReplyDelta))
        assert isinstance(events[-1], CompletedTurn)
        assert text == events[-1].reply == f"고소한 맛이 좋군요. {CONNECTED_QUESTION}"
        assert PREVIOUS_QUESTION not in text
        assert (await states.load(101)).last_user_message_id == 5

    asyncio.run(run())


def test_explicit_exit_finishes_without_generating_another_question():
    async def run():
        gateway = Gateway("이야기 들려줘서 고마워요.")
        use_cases, states, state = setup(gateway)
        await states.save(state)
        completed = await use_cases.send_message(101, "이제 그만할게요", user_id=1)
        assert "?" not in completed.reply
        assert len(gateway.calls) == 1
        assert completed.state.status == SessionStatus.INPUT_LOCKED

    asyncio.run(run())


@pytest.mark.parametrize(
    ("question", "answer", "subject", "confirmation"),
    [
        (PREVIOUS_QUESTION, "고소한 거요", "고소한 커피", 0.1),
        ("카페는 조용한 곳이 좋아요?", "네", "조용한 카페", 0.95),
    ],
)
def test_short_answer_is_grounded_in_context_and_saved_without_asking_for_a_reason(
    question, answer, subject, confirmation
):
    candidate = Candidate(subject, CandidateKind.ATTRIBUTE)
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.history = [ConversationTurn("assistant", question, utc_now())]
    request = build_judgment_request(state, answer, [candidate])
    assert request.state["previous_ai_message"] == question
    assert request.state["user_message"] == answer
    assert "confirmation_0" in request.questions
    delta, _ = judgment_delta(
        [candidate],
        {
            "stance_0": {"choice": "like", "confidence": 0.95},
            "material_0": {"choice": "taste", "confidence": 0.95},
            "explicit_0": {"noul": 0.95},
            "confirmation_0": {"noul": confirmation},
        },
        request.sentences,
    )
    merged = ProfileMerger().merge(
        ProfileState(), delta, utterance=answer, now=utc_now(), source_turn=1
    )
    (signal,) = merged.accepted
    assert signal.field == TasteField.PREFERENCES
    assert signal.value == subject
    assert signal.evidence == answer
    assert signal.evidence_type == (
        EvidenceType.CONFIRMED if confirmation >= 0.8 else EvidenceType.EXPLICIT
    )
    assert signal.reason is None


def test_yes_to_an_ai_suggested_worldview_is_not_saved_as_the_users_value():
    delta, _ = judgment_delta(
        [Candidate("독립적인 삶", CandidateKind.ATTRIBUTE, insight_kind=InsightKind.VALUE)],
        {
            "stance_0": {"choice": "like", "confidence": 0.95},
            "material_0": {"choice": "taste", "confidence": 0.95},
            "explicit_0": {"noul": 0.95},
            "confirmation_0": {"noul": 0.95},
            "meaning_0": {"noul": 0.95},
        },
        ["네"],
    )
    assert not delta.items
