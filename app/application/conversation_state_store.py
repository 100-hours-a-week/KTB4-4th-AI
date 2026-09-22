from __future__ import annotations

from datetime import datetime

from app.application.ports.session_store import SessionStore
from app.domain.conversation.models import ConversationState
from app.domain.profile.models import utc_now


class SessionLifetimeExceededError(Exception):
    """Raised when a conversation has reached its absolute lifetime."""


class ConversationStateStore:
    """Serialize conversation state behind a Redis-compatible storage port."""

    def __init__(
        self,
        store: SessionStore,
        *,
        idle_ttl_seconds: int = 1800,
        max_lifetime_seconds: int = 7200,
    ) -> None:
        if idle_ttl_seconds <= 0 or max_lifetime_seconds <= 0:
            raise ValueError("session TTL values must be positive")
        self._store = store
        self._idle_ttl_seconds = idle_ttl_seconds
        self._max_lifetime_seconds = max_lifetime_seconds

    def _ttl_seconds(self, state: ConversationState, now: datetime) -> int:
        age_seconds = max((now - state.created_at).total_seconds(), 0.0)
        remaining_lifetime = int(self._max_lifetime_seconds - age_seconds)
        if remaining_lifetime <= 0:
            raise SessionLifetimeExceededError(state.session_id)
        return min(self._idle_ttl_seconds, remaining_lifetime)

    async def load(
        self,
        session_id: int,
        *,
        now: datetime | None = None,
    ) -> ConversationState | None:
        payload = await self._store.load(session_id)
        if payload is None:
            return None
        state = ConversationState.from_dict(dict(payload))
        try:
            self._ttl_seconds(state, now or utc_now())
        except SessionLifetimeExceededError:
            return None
        return state

    async def save(
        self,
        state: ConversationState,
        *,
        now: datetime | None = None,
    ) -> None:
        ttl_seconds = self._ttl_seconds(state, now or utc_now())
        await self._store.save(
            state.session_id,
            state.to_dict(),
            ttl_seconds=ttl_seconds,
        )

    async def delete(self, session_id: int) -> bool:
        return await self._store.delete(session_id)
