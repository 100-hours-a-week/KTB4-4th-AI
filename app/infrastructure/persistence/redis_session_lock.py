from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import suppress

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# 다른 요청의 락을 지우거나 늘리지 않도록 토큰이 같을 때만 처리한다.
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""
_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""


class RedisSessionLease:
    def __init__(self, owner: RedisSessionLock, key: str, token: str) -> None:
        self._owner = owner
        self._key = key
        self._token = token
        self._lost = False
        self._heartbeat = asyncio.create_task(self._keep_alive())

    async def renew(self) -> bool:
        if self._lost:
            return False
        renewed = await self._owner.renew(self._key, self._token)
        if not renewed:
            self._lost = True
            logger.warning("session lock lost: key=%s", self._key)
        return renewed

    async def release(self) -> None:
        self._heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await self._heartbeat
        if self._lost:
            return
        try:
            await self._owner.release(self._key, self._token)
        except Exception:
            # 풀지 못해도 TTL이 지나면 풀린다. 요청 결과는 이미 정해졌으므로 실패로 바꾸지 않는다.
            logger.warning("session lock release failed: key=%s", self._key, exc_info=True)

    async def _keep_alive(self) -> None:
        while not self._lost:
            await asyncio.sleep(self._owner.heartbeat_seconds)
            try:
                await self.renew()
            except Exception:
                # 한 번 실패해도 TTL이 남아 있으면 다음 연장으로 이어진다.
                # 락을 잃었는지는 저장 직전 확인으로 판단한다.
                logger.warning("session lock renew failed: key=%s", self._key, exc_info=True)


class RedisSessionLock:
    """AI 서버가 여러 대여도 같은 대화 세션에는 한 요청만 들어가게 한다.

    잡은 쪽이 죽으면 TTL이 지나 풀리고, 살아 있는 동안에는 heartbeat 로 TTL을 늘린다.
    """

    def __init__(
        self,
        client: Redis,
        *,
        ttl_seconds: float = 30.0,
        poll_interval_seconds: float = 0.1,
        key_prefix: str = "needu:chat:session-lock:",
    ) -> None:
        if ttl_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("session lock intervals must be positive")
        self._client = client
        self._ttl_ms = max(1, round(ttl_seconds * 1000))
        # TTL 안에 세 번 연장을 시도해 한두 번 늦거나 실패해도 풀리지 않게 한다.
        self.heartbeat_seconds = ttl_seconds / 3
        self._poll_interval_seconds = poll_interval_seconds
        self._key_prefix = key_prefix
        self._release_script = client.register_script(_RELEASE_SCRIPT)
        self._renew_script = client.register_script(_RENEW_SCRIPT)

    def _key(self, session_id: int) -> str:
        return f"{self._key_prefix}{session_id}"

    async def acquire(self, session_id: int, *, wait_seconds: float) -> RedisSessionLease | None:
        key = self._key(session_id)
        token = uuid.uuid4().hex
        deadline = time.monotonic() + max(wait_seconds, 0.0)
        while True:
            if await self._client.set(key, token, nx=True, px=self._ttl_ms):
                return RedisSessionLease(self, key, token)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(self._poll_interval_seconds, remaining))

    async def is_locked(self, session_id: int) -> bool:
        return bool(await self._client.exists(self._key(session_id)))

    async def renew(self, key: str, token: str) -> bool:
        return bool(await self._renew_script(keys=[key], args=[token, self._ttl_ms]))

    async def release(self, key: str, token: str) -> None:
        await self._release_script(keys=[key], args=[token])
