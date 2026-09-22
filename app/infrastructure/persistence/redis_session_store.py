from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from redis.asyncio import Redis


class RedisSessionStore:
    def __init__(self, client: Redis, *, key_prefix: str = "needu:chat:session:") -> None:
        self._client = client
        self._key_prefix = key_prefix

    def _key(self, session_id: int) -> str:
        return f"{self._key_prefix}{session_id}"

    async def load(self, session_id: int) -> Mapping[str, Any] | None:
        raw = await self._client.get(self._key(session_id))
        if raw is None:
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        return payload

    async def save(
        self,
        session_id: int,
        state: Mapping[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        await self._client.set(
            self._key(session_id),
            json.dumps(dict(state), ensure_ascii=False, separators=(",", ":")),
            ex=ttl_seconds,
        )

    async def delete(self, session_id: int) -> bool:
        return bool(await self._client.delete(self._key(session_id)))
