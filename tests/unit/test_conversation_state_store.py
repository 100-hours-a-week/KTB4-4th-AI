import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.application.conversation_state_store import (
    ConversationStateStore,
    SessionLifetimeExceededError,
)
from app.domain.conversation.models import ConversationGoal, ConversationState, ConversationTurn


class FakeSessionStore:
    def __init__(self) -> None:
        self.values: dict[int, Mapping[str, Any]] = {}
        self.last_ttl_seconds: int | None = None

    async def load(self, session_id: int) -> Mapping[str, Any] | None:
        return self.values.get(session_id)

    async def save(
        self,
        session_id: int,
        state: Mapping[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        self.values[session_id] = state
        self.last_ttl_seconds = ttl_seconds

    async def delete(self, session_id: int) -> bool:
        return self.values.pop(session_id, None) is not None


def test_state_store_round_trips_redis_serializable_state() -> None:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    adapter = FakeSessionStore()
    store = ConversationStateStore(adapter)
    state = ConversationState(
        user_id=1,
        conversation_room_id=101,
        last_goal=ConversationGoal.INTEREST,
        history=[ConversationTurn(role="assistant", content="안녕하세요", created_at=now)],
        created_at=now,
        last_active_at=now,
    )

    asyncio.run(store.save(state, now=now))
    loaded = asyncio.run(store.load(101, now=now + timedelta(minutes=1)))

    assert adapter.last_ttl_seconds == 1800
    assert loaded is not None
    assert loaded.expiration_at == now + timedelta(minutes=30)
    assert loaded.to_dict() == state.to_dict()


def test_state_store_loads_legacy_messages_without_created_at() -> None:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    adapter = FakeSessionStore()
    store = ConversationStateStore(adapter)
    state = ConversationState(
        user_id=1,
        conversation_room_id=101,
        history=[ConversationTurn(role="assistant", content="안녕하세요", created_at=now)],
        created_at=now,
        last_active_at=now,
    )
    payload = state.to_dict()
    del payload["history"][0]["created_at"]
    adapter.values[101] = payload

    loaded = asyncio.run(store.load(101, now=now + timedelta(minutes=1)))

    assert loaded is not None
    assert loaded.history[0].created_at == now


def test_state_store_caps_idle_ttl_at_absolute_session_lifetime() -> None:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    adapter = FakeSessionStore()
    store = ConversationStateStore(adapter)
    state = ConversationState(
        user_id=1,
        conversation_room_id=101,
        created_at=now - timedelta(seconds=7100),
        last_active_at=now,
    )

    asyncio.run(store.save(state, now=now))

    assert adapter.last_ttl_seconds == 100
    assert state.expiration_at == now + timedelta(seconds=100)

    with pytest.raises(SessionLifetimeExceededError):
        asyncio.run(store.save(state, now=now + timedelta(seconds=101)))


def test_non_message_save_does_not_extend_idle_expiration() -> None:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    adapter = FakeSessionStore()
    store = ConversationStateStore(adapter)
    state = ConversationState(
        user_id=1,
        conversation_room_id=101,
        created_at=now,
        last_active_at=now,
    )

    expiration_at = asyncio.run(store.save(state, now=now + timedelta(minutes=10)))

    assert expiration_at == now + timedelta(minutes=30)
    assert adapter.last_ttl_seconds == 1200
