import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from redis.asyncio import Redis

from app.infrastructure.persistence import InMemorySessionLock, RedisSessionLock
from app.infrastructure.persistence.redis_session_lock import _RELEASE_SCRIPT, _RENEW_SCRIPT

# 설정하면 같은 테스트를 실제 Redis에서도 돌려 Lua 스크립트까지 확인한다.
REDIS_TEST_URL = os.environ.get("REDIS_TEST_URL")
TTL_SECONDS = 0.3


class FakeRedis:
    """락이 쓰는 명령만 흉내 낸다. 스크립트는 같은 동작을 파이썬으로 실행한다."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float]] = {}

    def _get(self, key: str) -> str | None:
        stored = self._values.get(key)
        if stored is None:
            return None
        value, expires_at = stored
        if time.monotonic() >= expires_at:
            del self._values[key]
            return None
        return value

    async def set(self, key: str, value: str, *, nx: bool = False, px: int) -> bool:
        if nx and self._get(key) is not None:
            return False
        self._values[key] = (value, time.monotonic() + px / 1000)
        return True

    async def exists(self, key: str) -> int:
        return int(self._get(key) is not None)

    def register_script(self, script: str) -> Callable[..., Any]:
        async def release(*, keys: list[str], args: list[Any]) -> int:
            if self._get(keys[0]) != args[0]:
                return 0
            del self._values[keys[0]]
            return 1

        async def renew(*, keys: list[str], args: list[Any]) -> int:
            if self._get(keys[0]) != args[0]:
                return 0
            self._values[keys[0]] = (args[0], time.monotonic() + int(args[1]) / 1000)
            return 1

        return {_RELEASE_SCRIPT: release, _RENEW_SCRIPT: renew}[script]


_BACKENDS = ["fake"] + (["redis"] if REDIS_TEST_URL else [])


@asynccontextmanager
async def redis_locks(backend: str, count: int = 2) -> AsyncIterator[list[RedisSessionLock]]:
    """AI 서버 여러 대처럼 클라이언트를 따로 둔 락들. 같은 Redis 키를 본다."""
    prefix = f"test:session-lock:{uuid.uuid4().hex}:"
    if backend == "fake":
        fake = FakeRedis()
        yield [
            RedisSessionLock(fake, ttl_seconds=TTL_SECONDS, key_prefix=prefix)  # type: ignore[arg-type]
            for _ in range(count)
        ]
        return
    clients = [Redis.from_url(REDIS_TEST_URL, decode_responses=True) for _ in range(count)]
    try:
        yield [
            RedisSessionLock(client, ttl_seconds=TTL_SECONDS, key_prefix=prefix)
            for client in clients
        ]
    finally:
        for client in clients:
            await client.aclose()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_other_server_cannot_take_held_session(backend: str) -> None:
    async def scenario() -> None:
        async with redis_locks(backend) as (first, second):
            lease = await first.acquire(1, wait_seconds=0)
            assert lease is not None
            assert await second.acquire(1, wait_seconds=0) is None
            assert await second.is_locked(1)
            # 다른 세션은 막지 않는다.
            other = await second.acquire(2, wait_seconds=0)
            assert other is not None

            await lease.release()
            await other.release()
            assert not await second.is_locked(1)
            retaken = await second.acquire(1, wait_seconds=0)
            assert retaken is not None
            await retaken.release()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_waiting_request_gets_lock_after_release(backend: str) -> None:
    async def scenario() -> None:
        async with redis_locks(backend) as (first, second):
            lease = await first.acquire(1, wait_seconds=0)
            assert lease is not None
            waiting = asyncio.create_task(second.acquire(1, wait_seconds=1))
            await asyncio.sleep(0.15)
            assert not waiting.done()
            await lease.release()
            acquired = await waiting
            assert acquired is not None
            await acquired.release()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_waiting_request_gives_up_after_wait_seconds(backend: str) -> None:
    async def scenario() -> None:
        async with redis_locks(backend) as (first, second):
            lease = await first.acquire(1, wait_seconds=0)
            assert lease is not None
            started = time.monotonic()
            assert await second.acquire(1, wait_seconds=0.15) is None
            assert time.monotonic() - started >= 0.15
            await lease.release()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_holder_keeps_lock_past_ttl_while_alive(backend: str) -> None:
    async def scenario() -> None:
        async with redis_locks(backend) as (first, second):
            lease = await first.acquire(1, wait_seconds=0)
            assert lease is not None
            await asyncio.sleep(TTL_SECONDS * 3)
            assert await second.acquire(1, wait_seconds=0) is None
            assert await lease.renew()
            await lease.release()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_lock_of_dead_server_expires_and_late_release_keeps_new_owner(backend: str) -> None:
    async def scenario() -> None:
        async with redis_locks(backend) as (first, second):
            lease = await first.acquire(1, wait_seconds=0)
            assert lease is not None
            # 서버가 멈춰 연장하지 못한 상황
            lease._heartbeat.cancel()
            await asyncio.sleep(TTL_SECONDS + 0.1)

            taken = await second.acquire(1, wait_seconds=0)
            assert taken is not None
            # 늦게 깨어난 쪽은 저장 전 확인에서 막히고, 해제해도 새 주인의 락을 지우지 않는다.
            assert not await lease.renew()
            await lease.release()
            assert await second.is_locked(1)
            assert await taken.renew()
            await taken.release()

    asyncio.run(scenario())


def test_in_memory_lock_rejects_and_waits_like_redis_lock() -> None:
    async def scenario() -> None:
        lock = InMemorySessionLock()
        lease = await lock.acquire(1, wait_seconds=0)
        assert lease is not None
        assert await lock.is_locked(1)
        assert await lock.acquire(1, wait_seconds=0) is None
        assert await lock.acquire(1, wait_seconds=0.05) is None

        waiting = asyncio.create_task(lock.acquire(1, wait_seconds=1))
        await asyncio.sleep(0.01)
        await lease.release()
        await lease.release()
        assert not await lease.renew()
        acquired = await waiting
        assert acquired is not None
        assert await acquired.renew()
        await acquired.release()
        assert not await lock.is_locked(1)

    asyncio.run(scenario())
