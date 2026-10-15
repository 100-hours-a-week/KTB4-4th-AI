import asyncio
import json
import random
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes import v1_chat
from app.application.chat_use_cases import ChatUseCases, ReplyDelta
from app.application.conversation_service import CompletedTurn, ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.ports.model_gateway import Message
from app.core.config import Settings
from app.domain.conversation.guards import (
    CLOSING_MESSAGE,
    closing_reply,
    sanitize_response,
    stable_reply_prefix,
)
from app.domain.conversation.models import ConversationGoal, ConversationState, SessionStatus
from app.infrastructure.persistence import InMemorySessionStore
from app.main import create_app

HEADERS = {"Authorization": "Bearer test-token"}
EXTRACTION = {
    "items": [
        {
            "field": "hobbies",
            "value": "캠핑",
            "confidence": 0.9,
            "evidence": "주말마다 캠핑 가요",
            "evidenceType": "explicit",
            "intentType": "both",
            "deferralReason": None,
        }
    ],
    "axes": [],
    "drop": [],
}


class StreamGateway:
    def __init__(
        self,
        *,
        chunks: list[str] | None = None,
        fail: bool = False,
        chunk_delay: float = 0.0,
    ) -> None:
        self.chunks = chunks or ["캠핑 좋죠. ", "주로 어디로", " 다니세요?"]
        self.fail = fail
        self.chunk_delay = chunk_delay
        self.stream_calls = 0
        self.extraction_gate: asyncio.Event | None = None

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        return "안녕하세요! 요즘 어떻게 지내세요?"

    async def stream(self, messages: Sequence[Message], *, model: str) -> AsyncIterator[str]:
        self.stream_calls += 1
        if self.fail:
            raise RuntimeError("upstream down")
        for chunk in self.chunks:
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield chunk

    async def structured(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.extraction_gate is not None:
            await self.extraction_gate.wait()
        return EXTRACTION


def make_app(gateway: StreamGateway):
    return create_app(
        Settings(
            app_env="test",
            service_token="test-token",
            redis_url=None,
            response_model_base_url=None,
            extraction_model_base_url=None,
        ),
        model_gateway=gateway,
        session_store=InMemorySessionStore(),
    )


def parse_sse(text: str) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    events: list[tuple[str, dict[str, Any]]] = []
    pings = 0
    for block in text.strip().split("\n\n"):
        if block == ": ping":
            pings += 1
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events, pings


def start_session(client: TestClient, room_id: int = 9001) -> None:
    created = client.post(
        "/v1/chat/sessions",
        headers=HEADERS,
        json={"userId": 1, "conversationRoomId": room_id},
    )
    assert created.status_code == 201


def stream(client: TestClient, user_message_id: int, *, room_id: int = 9001):
    return client.post(
        f"/v1/chat/sessions/{room_id}/messages/stream",
        headers={**HEADERS, "Accept": "text/event-stream"},
        json={"userId": 1, "userMessageId": user_message_id, "message": "주말마다 캠핑 가요"},
    )


def test_stream_sends_deltas_that_join_into_completed_reply() -> None:
    gateway = StreamGateway()
    with TestClient(make_app(gateway)) as client:
        start_session(client)
        response = stream(client, 1)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events, _ = parse_sse(response.text)
    deltas = [data for name, data in events if name == "delta"]
    assert [delta["index"] for delta in deltas] == list(range(len(deltas)))
    assert events[-1][0] == "completed"
    completed = events[-1][1]
    assert "".join(delta["text"] for delta in deltas) == completed["reply"]
    assert completed["reply"] == "캠핑 좋죠. 주로 어디로 다니세요?"
    # 문장이 끝나기를 기다리지 않고 받은 만큼 바로 보낸다.
    assert [delta["text"] for delta in deltas] == ["캠핑 좋죠.", " 주로 어디로", " 다니세요?"]
    assert completed["turn"] == 1
    assert completed["inputLocked"] is False
    assert 0 <= completed["progress"] <= 100
    assert completed["expirationAt"]


def test_same_user_message_returns_saved_turn_without_generating() -> None:
    gateway = StreamGateway()
    with TestClient(make_app(gateway)) as client:
        start_session(client)
        first = stream(client, 1)
        again = stream(client, 1)

    first_events, _ = parse_sse(first.text)
    again_events, _ = parse_sse(again.text)
    assert gateway.stream_calls == 1
    assert [name for name, _ in again_events] == ["completed"]
    assert again_events[0][1] == first_events[-1][1]


def test_confirmed_message_is_returned_even_after_input_locks() -> None:
    gateway = StreamGateway(chunks=["네, 오늘 즐거웠어요!"])
    app = make_app(gateway)
    with TestClient(app) as client:
        start_session(client)
        store = app.state.conversation_state_store
        state = asyncio.run(store.load(9001))
        state.turn_count = 19
        asyncio.run(store.save(state))
        last = stream(client, 20)
        again = stream(client, 20)
        locked = stream(client, 21)

    last_events, _ = parse_sse(last.text)
    assert last_events[-1][1]["inputLocked"] is True
    assert last_events[-1][1]["reply"] == f"네, 오늘 즐거웠어요! {CLOSING_MESSAGE}"
    again_events, _ = parse_sse(again.text)
    assert [name for name, _ in again_events] == ["completed"]
    assert locked.status_code == 409
    assert locked.json()["code"] == "INPUT_LOCKED"


def test_model_failure_sends_error_and_keeps_turn_unsaved() -> None:
    gateway = StreamGateway(fail=True)
    app = make_app(gateway)
    with TestClient(app) as client:
        start_session(client)
        failed = stream(client, 1)
        gateway.fail = False
        retried = stream(client, 1)

    events, _ = parse_sse(failed.text)
    assert events == [
        (
            "error",
            {
                "code": "LLM_UNAVAILABLE",
                "message": "대화 모델을 호출할 수 없습니다.",
                "retryable": True,
            },
        )
    ]
    retried_events, _ = parse_sse(retried.text)
    assert retried_events[-1][0] == "completed"
    assert retried_events[-1][1]["turn"] == 1


def test_requests_rejected_before_stream_use_http_errors() -> None:
    gateway = StreamGateway()
    with TestClient(make_app(gateway)) as client:
        missing_session = stream(client, 1, room_id=404)
        start_session(client)
        missing_id = client.post(
            "/v1/chat/sessions/9001/messages/stream",
            headers=HEADERS,
            json={"userId": 1, "message": "주말마다 캠핑 가요"},
        )

    assert missing_session.status_code == 404
    assert missing_session.json()["code"] == "SESSION_NOT_FOUND"
    assert missing_id.status_code == 400
    assert missing_id.json()["code"] == "INVALID_REQUEST"


def test_stream_sends_ping_while_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v1_chat, "STREAM_PING_INTERVAL_SECONDS", 0.01)
    gateway = StreamGateway(chunks=["캠핑 좋죠?"], chunk_delay=0.05)
    with TestClient(make_app(gateway)) as client:
        start_session(client)
        response = stream(client, 1)

    events, pings = parse_sse(response.text)
    assert pings >= 1
    assert events[-1][0] == "completed"


def _use_cases(gateway: StreamGateway) -> tuple[ChatUseCases, ConversationStateStore]:
    states = ConversationStateStore(InMemorySessionStore())
    conversations = ConversationService(model_gateway=gateway, extraction_model="extractor")
    use_cases = ChatUseCases(
        conversations=conversations,
        states=states,
        model_gateway=gateway,
        response_model="responder",
    )
    return use_cases, states


def test_question_streams_before_extraction_finishes() -> None:
    async def scenario() -> None:
        gateway = StreamGateway()
        gateway.extraction_gate = asyncio.Event()
        use_cases, states = _use_cases(gateway)
        await use_cases.create_session(user_id=1, conversation_room_id=9001)

        events = use_cases.stream_message(9001, "주말마다 캠핑 가요", user_id=1, user_message_id=1)
        streamed = ""
        async with asyncio.timeout(1):
            while "?" not in streamed:
                item = await anext(events)
                assert isinstance(item, ReplyDelta)
                streamed += item.text
        # 추출이 끝나지 않았어도 질문까지 이미 나갔다.
        assert streamed == "캠핑 좋죠. 주로 어디로 다니세요?"
        gateway.extraction_gate.set()
        rest = [item async for item in events]
        assert isinstance(rest[-1], CompletedTurn)
        assert rest[-1].reply == streamed

    asyncio.run(scenario())


def test_closed_stream_cancels_generation_and_saves_nothing() -> None:
    async def scenario() -> None:
        gateway = StreamGateway()
        gateway.extraction_gate = asyncio.Event()
        use_cases, states = _use_cases(gateway)
        await use_cases.create_session(user_id=1, conversation_room_id=9001)

        sse = v1_chat._sse_stream(
            use_cases.stream_message(9001, "주말마다 캠핑 가요", user_id=1, user_message_id=1)
        )
        first = await anext(sse)
        assert first.startswith("event: delta")
        await sse.aclose()

        state = await states.load(9001)
        assert state.turn_count == 0
        assert state.last_user_message_id is None
        assert not use_cases._lock(9001).locked()

    asyncio.run(scenario())


def _final_reply(raw: str, *, wrap: bool) -> str:
    goal = ConversationGoal.WRAP if wrap else ConversationGoal.CHAT
    reply = sanitize_response(raw, goal)
    return closing_reply(reply) if wrap else reply


@pytest.mark.parametrize("wrap", [False, True])
def test_stable_prefix_is_always_a_prefix_of_final_reply(wrap: bool) -> None:
    pieces = [
        "니쥬: ",
        "좋아요",
        ". ",
        "진짜요?",
        " 대박",
        "!",
        " 주로 뭐 해요",
        "?",
        " 저는 라면요.",
        "\n",
        "그리고",
        " 이건 아주 길게 늘어놓는 말이라서 서른 글자를 훌쩍 넘어갑니다.",
        "~",
        "사용자:",
        "사용",
        "자",
        "[예",
        "시]",
        "현재 ",
        "목표:",
        " 끝",
        '"',
        "니",
        "쥬",
        ":",
        "AI",
        " ",
        "  ",
    ]
    rng = random.Random(7)
    for _ in range(3000):
        raw = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
        final = _final_reply(raw, wrap=wrap)
        emitted = ""
        for end in range(1, len(raw) + 1):
            stable = stable_reply_prefix(raw[:end], hold_question=wrap)
            if len(stable) > len(emitted) and stable.startswith(emitted):
                emitted = stable
        assert final.startswith(emitted), (raw, emitted, final)


def test_stable_prefix_rules() -> None:
    # 일반 턴은 생성된 글자를 문장이 끝나기 전에도 바로 낸다.
    assert stable_reply_prefix("캠핑 좋", hold_question=False) == "캠핑 좋"
    assert stable_reply_prefix("좋아요. 뭐 해요?", hold_question=False) == "좋아요. 뭐 해요?"
    # 끝 공백과 줄바꿈은 다음 글자를 볼 때까지 보류한다.
    assert stable_reply_prefix("좋아요.\n", hold_question=False) == "좋아요."
    # 끝에 걸린 유출 표시의 앞부분은 보류하고, 표시가 완성되면 그 앞까지만 낸다.
    assert stable_reply_prefix("좋아요. 사용", hold_question=False) == "좋아요."
    assert stable_reply_prefix("좋아요. 사용자: 저는", hold_question=False) == "좋아요."
    assert stable_reply_prefix("좋아요. 사용해", hold_question=False) == "좋아요. 사용해"
    # 마지막 질문 뒤의 말은 길면 잘리므로 보류한다.
    assert stable_reply_prefix("뭐 해요? 저는", hold_question=False) == "뭐 해요?"
    # 이름표가 될 수 있는 앞부분은 보류하고, 이름표는 지운다.
    assert stable_reply_prefix("니쥬", hold_question=False) == ""
    assert stable_reply_prefix("니쥬: 좋아요. 그", hold_question=False) == "좋아요. 그"
    assert stable_reply_prefix("니쥬야 안녕", hold_question=False) == "니쥬야 안녕"
    # 따옴표로 시작하면 끝까지 보류한다.
    assert stable_reply_prefix('"좋아요. 그', hold_question=False) == ""
    # 마무리 턴은 끝 질문 문장을 보류하고 끝난 문장까지만 낸다.
    assert stable_reply_prefix("좋아요. 뭐 해요?", hold_question=True) == "좋아요."
    assert stable_reply_prefix("좋아요. 뭐", hold_question=True) == "좋아요."


def test_state_round_trip_keeps_stream_fields() -> None:
    from app.domain.conversation.models import CompletionReason

    state = ConversationState(user_id=1, conversation_room_id=9001)
    state.pending_wrap = CompletionReason.SUFFICIENT
    state.last_user_message_id = 701

    restored = ConversationState.from_dict(state.to_dict())

    assert restored.pending_wrap == CompletionReason.SUFFICIENT
    assert restored.last_user_message_id == 701
    assert restored.status == SessionStatus.ACTIVE
