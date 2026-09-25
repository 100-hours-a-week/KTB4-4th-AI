from __future__ import annotations

import copy
from collections.abc import Mapping
from time import monotonic
from typing import Any


class InMemorySessionStore:
    """Process-local session storage for tests and single-process local development."""

    def __init__(self) -> None:
        self._values: dict[int, tuple[float, dict[str, Any]]] = {}

    async def load(self, session_id: int) -> Mapping[str, Any] | None:
        stored = self._values.get(session_id)
        if stored is None:
            return None
        expires_at, value = stored
        if monotonic() >= expires_at:
            self._values.pop(session_id, None)
            return None
        return copy.deepcopy(value)

    async def save(
        self,
        session_id: int,
        state: Mapping[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        self._values[session_id] = (monotonic() + ttl_seconds, copy.deepcopy(dict(state)))

    async def delete(self, session_id: int) -> bool:
        return self._values.pop(session_id, None) is not None
