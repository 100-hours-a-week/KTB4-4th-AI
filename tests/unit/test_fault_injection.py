import asyncio

import pytest
from fastapi.testclient import TestClient

from app.application import chat_use_cases
from app.application.ports.fault_injection import ReplyDelayConfig, TestAccountConfig
from app.core.config import Settings
from app.domain.conversation.models import (
    ConversationGoal,
    ConversationState,
    SessionStatus,
)
from app.domain.conversation.policy import MAX_TURNS, decide_goal
from app.infrastructure.persistence import InMemorySessionStore
from app.infrastructure.persistence.redis_fault_injection import (
    InMemoryReplyDelayStore,
    _decode,
    _decode_accounts,
    _encode,
    _encode_accounts,
)
from app.main import create_app
from tests.unit.test_chat_stream import HEADERS, StreamGateway, parse_sse, start_session, stream

URL = "/v1/internal/fault/reply-delay"
ACCOUNTS_URL = "/v1/internal/fault/test-accounts"


def make_app(*, enabled: bool = True, gateway: StreamGateway | None = None):
    return create_app(
        Settings(
            app_env="test",
            service_token="test-token",
            redis_url=None,
            response_model_base_url=None,
            extraction_model_base_url=None,
            fault_injection_enabled=enabled,
            fault_injection_max_delay_seconds=5.0,
        ),
        model_gateway=gateway or StreamGateway(),
        session_store=InMemorySessionStore(),
    )


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """응답 지연으로 건 sleep만 모은다. 다른 sleep은 그대로 둔다."""
    recorded: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *args, **kwargs):
        if delay >= 1:
            recorded.append(delay)
            return None
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(chat_use_cases.asyncio, "sleep", fake_sleep)
    return recorded


def test_config_delays_only_targets() -> None:
    config = ReplyDelayConfig(delay_seconds=3, user_ids=frozenset({1}), session_ids=frozenset({7}))
    assert config.delay_for(session_id=99, user_id=1) == 3
    assert config.delay_for(session_id=7, user_id=2) == 3
    assert config.delay_for(session_id=99, user_id=2) == 0


def test_redis_format_round_trip() -> None:
    config = ReplyDelayConfig(delay_seconds=3, user_ids=frozenset({2, 1}))
    assert _decode(_encode(config)) == config
    # redis-cli로 일부 필드만 넣어도 읽는다.
    assert _decode('{"delay_seconds": 3, "session_ids": [5]}') == ReplyDelayConfig(
        delay_seconds=3, session_ids=frozenset({5})
    )


def test_memory_store_expires() -> None:
    store = InMemoryReplyDelayStore()
    asyncio.run(store.set(ReplyDelayConfig(delay_seconds=3), ttl_seconds=0))
    assert asyncio.run(store.get()) is None


def test_target_user_reply_is_delayed(sleeps: list[float]) -> None:
    with TestClient(make_app()) as client:
        put = client.put(URL, headers=HEADERS, json={"delaySeconds": 3, "userIds": [1]})
        assert put.status_code == 200
        assert put.json()["active"] is True
        assert put.json()["expiresInSeconds"] > 0
        start_session(client)
        response = stream(client, 1)

    events, _ = parse_sse(response.text)
    assert events[-1][0] == "completed"
    assert sleeps == [3]


def test_other_user_reply_is_not_delayed(sleeps: list[float]) -> None:
    with TestClient(make_app()) as client:
        client.put(URL, headers=HEADERS, json={"delaySeconds": 3, "userIds": [2]})
        start_session(client)
        response = stream(client, 1)

    assert parse_sse(response.text)[0][-1][0] == "completed"
    assert sleeps == []


def test_delete_turns_delay_off(sleeps: list[float]) -> None:
    with TestClient(make_app()) as client:
        client.put(URL, headers=HEADERS, json={"delaySeconds": 3, "sessionIds": [9001]})
        assert client.delete(URL, headers=HEADERS).status_code == 204
        assert client.get(URL, headers=HEADERS).json()["active"] is False
        start_session(client)
        stream(client, 1)

    assert sleeps == []


def test_delay_is_capped_by_settings(sleeps: list[float]) -> None:
    app = make_app()
    with TestClient(app) as client:
        # API를 거치지 않고 저장소에 직접 넣은 값도 최대값으로 자른다.
        asyncio.run(
            app.state.reply_delay_store.set(
                ReplyDelayConfig(delay_seconds=30, user_ids=frozenset({1})), ttl_seconds=60
            )
        )
        start_session(client)
        stream(client, 1)

    assert sleeps == [5.0]


def test_store_failure_does_not_delay(sleeps: list[float]) -> None:
    app = make_app()

    async def broken_get():
        raise ConnectionError("redis down")

    with TestClient(app) as client:
        app.state.reply_delay_store.get = broken_get
        start_session(client)
        response = stream(client, 1)

    assert parse_sse(response.text)[0][-1][0] == "completed"
    assert sleeps == []


def test_put_requires_target_and_token() -> None:
    with TestClient(make_app()) as client:
        assert client.put(URL, headers=HEADERS, json={"delaySeconds": 3}).status_code == 422
        unauthorized = client.put(URL, json={"delaySeconds": 3, "userIds": [1]})
        assert unauthorized.status_code == 401


def test_disabled_server_ignores_api(sleeps: list[float]) -> None:
    app = make_app(enabled=False)
    with TestClient(app) as client:
        response = client.put(URL, headers=HEADERS, json={"delaySeconds": 3, "userIds": [1]})
        assert response.status_code == 404
        assert response.json()["code"] == "FAULT_INJECTION_DISABLED"
        assert app.state.reply_delay_store is None
        start_session(client)
        stream(client, 1)

    assert sleeps == []


def waived_state(turn_count: int = 0) -> ConversationState:
    state = ConversationState(user_id=1, conversation_room_id=101, turn_count=turn_count)
    state.turn_limits_waived = True
    return state


def test_waived_state_survives_round_trip() -> None:
    assert ConversationState.from_dict(waived_state().to_dict()).turn_limits_waived is True
    # 이전 버전에서 저장한 세션은 일반 정책이다.
    payload = ConversationState(user_id=1, conversation_room_id=101).to_dict()
    payload.pop("turn_limits_waived")
    assert ConversationState.from_dict(payload).turn_limits_waived is False


def test_waived_state_skips_wrap_conditions() -> None:
    normal = ConversationState(user_id=1, conversation_room_id=101, turn_count=MAX_TURNS - 1)
    assert decide_goal(normal, "주말마다 캠핑 가요").goal == ConversationGoal.WRAP
    assert decide_goal(waived_state(MAX_TURNS + 30), "주말마다 캠핑 가요").goal != (
        ConversationGoal.WRAP
    )
    assert decide_goal(waived_state(), "이제 그만할게요").goal != ConversationGoal.WRAP
    tired = waived_state()
    tired.low_energy_streak = 10
    assert decide_goal(tired, "네").goal != ConversationGoal.WRAP


def test_test_account_keeps_chatting_past_max_turns() -> None:
    gateway = StreamGateway()
    with TestClient(make_app(gateway=gateway)) as client:
        put = client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [1], "mockModel": False})
        assert put.json()["active"] is True
        assert put.json()["mockModel"] is False
        assert put.json()["userCount"] == 1
        assert 7100 < put.json()["expiresInSeconds"] <= 7200
        start_session(client)
        # 같은 문장을 최대 턴보다 많이 보내도 매번 모델까지 가고 입력이 잠기지 않는다.
        turns = MAX_TURNS + 5
        for user_message_id in range(1, turns + 1):
            events, _ = parse_sse(stream(client, user_message_id).text)
            assert events[-1][0] == "completed", events[-1]
            assert events[-1][1]["inputLocked"] is False

    assert events[-1][1]["turn"] == turns
    assert gateway.stream_calls == turns


def test_other_account_still_locks_at_max_turns() -> None:
    app = make_app()
    with TestClient(app) as client:
        client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [2]})
        start_session(client)
        store = app.state.conversation_state_store
        state = asyncio.run(store.load(9001))
        state.turn_count = MAX_TURNS - 1
        asyncio.run(store.save(state))
        events, _ = parse_sse(stream(client, 1).text)

    assert events[-1][1]["inputLocked"] is True


def test_test_account_ignores_model_end_mark() -> None:
    class EndingGateway(StreamGateway):
        ending = False

        async def complete(self, messages, *, model: str) -> str:
            # 첫인사는 정상으로 만들고, 그 뒤 질문 보완에서만 니쥬가 대화를 마친다.
            return "종료" if self.ending else await super().complete(messages, model=model)

    gateway = EndingGateway(chunks=["캠핑 좋죠."])
    app = make_app(gateway=gateway)
    with TestClient(app) as client:
        client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [1], "mockModel": False})
        start_session(client)
        gateway.ending = True
        events, _ = parse_sse(stream(client, 1).text)
        state = asyncio.run(app.state.conversation_state_store.load(9001))

    assert events[-1][1]["inputLocked"] is False
    assert state.status == SessionStatus.ACTIVE


def test_policy_stays_after_account_list_is_cleared() -> None:
    app = make_app()
    with TestClient(app) as client:
        client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [1]})
        start_session(client)
        stream(client, 1)
        assert client.delete(ACCOUNTS_URL, headers=HEADERS).status_code == 204
        assert client.get(ACCOUNTS_URL, headers=HEADERS).json()["active"] is False
        state = asyncio.run(app.state.conversation_state_store.load(9001))

    assert state.turn_limits_waived is True


def test_account_list_failure_keeps_normal_policy() -> None:
    app = make_app()

    async def broken_get():
        raise ConnectionError("redis down")

    with TestClient(app) as client:
        app.state.test_account_store.get = broken_get
        start_session(client)
        stream(client, 1)
        state = asyncio.run(app.state.conversation_state_store.load(9001))

    assert state.turn_limits_waived is False


def test_test_accounts_api_validation() -> None:
    with TestClient(make_app()) as client:
        assert client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": []}).status_code == 400
        assert client.put(ACCOUNTS_URL, json={"userIds": [1]}).status_code == 401
    with TestClient(make_app(enabled=False)) as client:
        disabled = client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [1]})
        assert disabled.status_code == 404


class ForbiddenGateway(StreamGateway):
    """Mock 세션에서 모델을 한 번이라도 부르면 실패한다."""

    async def complete(self, messages, *, model: str) -> str:
        raise AssertionError("model called")

    async def stream(self, messages, *, model: str):
        raise AssertionError("model called")
        yield ""

    async def structured(self, messages, *, model: str, json_schema):
        raise AssertionError("model called")


def test_account_config_round_trip() -> None:
    config = TestAccountConfig(user_ids=frozenset({3, 1}), mock_model=False)
    assert _decode_accounts(_encode_accounts(config)) == config
    # 직접 넣을 때 mock_model을 빠뜨리면 비용이 들지 않는 Mock으로 본다.
    assert _decode_accounts('{"user_ids": [1]}').mock_model is True


def test_mock_account_never_calls_model() -> None:
    gateway = ForbiddenGateway()
    app = make_app(gateway=gateway)
    with TestClient(app) as client:
        put = client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [1]})
        assert put.json()["mockModel"] is True
        created = client.post(
            "/v1/chat/sessions",
            headers=HEADERS,
            json={"userId": 1, "conversationRoomId": 9001},
        )
        assert created.status_code == 201, created.json()
        for user_message_id in range(1, MAX_TURNS + 6):
            events, _ = parse_sse(stream(client, user_message_id).text)
            assert events[-1][0] == "completed", events[-1]
            assert events[-1][1]["inputLocked"] is False
        sent = client.post(
            "/v1/chat/sessions/9001/messages",
            headers=HEADERS,
            json={"userId": 1, "message": "주말마다 캠핑 가요"},
        )
        assert sent.status_code == 200, sent.json()

    assert gateway.stream_calls == 0
    state = asyncio.run(app.state.conversation_state_store.load(9001))
    assert len(state.history) == chat_use_cases.TEST_SESSION_HISTORY_LIMIT
    assert state.turn_count == MAX_TURNS + 6
    completed = events[-1][1]
    deltas = [data["text"] for name, data in events if name == "delta"]
    assert "".join(deltas) == completed["reply"]
    assert completed["reply"].startswith("테스트 응답이에요.")
    assert completed["reply"].endswith("?")


def test_mock_reply_takes_injected_delay(sleeps: list[float]) -> None:
    with TestClient(make_app(gateway=ForbiddenGateway())) as client:
        client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [1]})
        client.put(URL, headers=HEADERS, json={"delaySeconds": 3, "userIds": [1]})
        start_session(client)
        events, _ = parse_sse(stream(client, 1).text)
        # 지연을 끄면 바로 응답한다(적체 복구 구간).
        client.delete(URL, headers=HEADERS)
        stream(client, 2)

    assert events[-1][0] == "completed"
    assert sleeps == [3]


def test_mock_state_survives_round_trip() -> None:
    state = waived_state()
    state.mock_model_calls = True
    assert ConversationState.from_dict(state.to_dict()).mock_model_calls is True
