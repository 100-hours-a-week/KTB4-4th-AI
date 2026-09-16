from collections.abc import Mapping
from typing import Any, Protocol


class SessionStore(Protocol):
    async def load(self, session_id: str) -> Mapping[str, Any] | None: ...

    async def save(
        self,
        session_id: str,
        state: Mapping[str, Any],
        *,
        ttl_seconds: int,
    ) -> None: ...

    async def delete(self, session_id: str) -> bool: ...
