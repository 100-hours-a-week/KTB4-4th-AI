from __future__ import annotations

from datetime import datetime, timedelta
from math import ceil

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

    def expiration_at(self, state: ConversationState) -> datetime:
        idle_expiration = state.last_active_at + timedelta(seconds=self._idle_ttl_seconds)
        lifetime_expiration = state.created_at + timedelta(seconds=self._max_lifetime_seconds)
        return min(idle_expiration, lifetime_expiration)

    def _ttl_seconds(self, state: ConversationState, now: datetime) -> int:
        remaining_seconds = (self.expiration_at(state) - now).total_seconds()
        if remaining_seconds <= 0:
            raise SessionLifetimeExceededError(state.session_id)
        return ceil(remaining_seconds)

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
        state.expiration_at = self.expiration_at(state)
        return state

    async def save(
        self,
        state: ConversationState,
        *,
        now: datetime | None = None,
    ) -> datetime:
        resolved_now = now or utc_now()
        ttl_seconds = self._ttl_seconds(state, resolved_now)
        state.expiration_at = self.expiration_at(state)
        await self._store.save(
            state.session_id,
            state.to_dict(),
            ttl_seconds=ttl_seconds,
        )
        return state.expiration_at

    async def delete(self, session_id: int) -> bool:
        return await self._store.delete(session_id)
