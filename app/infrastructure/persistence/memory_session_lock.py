from __future__ import annotations

import asyncio


class InMemorySessionLease:
    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock
        self._released = False

    async def renew(self) -> bool:
        return not self._released

    async def release(self) -> None:
        if not self._released:
            self._released = True
            self._lock.release()


class InMemorySessionLock:
    """프로세스 안에서만 막는다. Redis 없이 띄우는 테스트와 로컬 단일 프로세스용."""

    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock(self, session_id: int) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def acquire(self, session_id: int, *, wait_seconds: float) -> InMemorySessionLease | None:
        lock = self._lock(session_id)
        if lock.locked() and wait_seconds <= 0:
            return None
        try:
            async with asyncio.timeout(wait_seconds if wait_seconds > 0 else None):
                await lock.acquire()
        except TimeoutError:
            return None
        return InMemorySessionLease(lock)

    async def is_locked(self, session_id: int) -> bool:
        return self._lock(session_id).locked()
