from __future__ import annotations

import json
import time

from redis.asyncio import Redis

from app.application.ports.fault_injection import ReplyDelayConfig, TestAccountConfig

REPLY_DELAY_KEY = "needu:fault:reply-delay"
TEST_ACCOUNTS_KEY = "needu:fault:test-accounts"


def _encode(config: ReplyDelayConfig) -> str:
    return json.dumps(
        {
            "delay_seconds": config.delay_seconds,
            "user_ids": sorted(config.user_ids),
            "session_ids": sorted(config.session_ids),
        }
    )


def _decode(raw: str) -> ReplyDelayConfig:
    body = json.loads(raw)
    return ReplyDelayConfig(
        delay_seconds=float(body.get("delay_seconds", 0)),
        user_ids=frozenset(int(value) for value in body.get("user_ids", [])),
        session_ids=frozenset(int(value) for value in body.get("session_ids", [])),
    )


class RedisReplyDelayStore:
    """응답 지연 설정을 Redis 키 하나에 둔다. redis-cli로 직접 넣어도 같은 형식이면 읽는다.

    SET needu:fault:reply-delay '{"delay_seconds":3,"user_ids":[1,2]}' EX 3600
    """

    def __init__(self, client: Redis, *, key: str = REPLY_DELAY_KEY) -> None:
        self._client = client
        self._key = key

    async def get(self) -> ReplyDelayConfig | None:
        raw = await self._client.get(self._key)
        return None if raw is None else _decode(raw)

    async def set(self, config: ReplyDelayConfig, *, ttl_seconds: int) -> None:
        await self._client.set(self._key, _encode(config), ex=ttl_seconds)

    async def ttl_seconds(self) -> int | None:
        ttl = await self._client.ttl(self._key)
        return ttl if ttl >= 0 else None

    async def clear(self) -> None:
        await self._client.delete(self._key)


class InMemoryReplyDelayStore:
    """Redis 없이 띄우는 로컬·테스트용. 프로세스 하나에서만 보인다."""

    def __init__(self) -> None:
        self._config: ReplyDelayConfig | None = None
        self._expires_at = 0.0

    def _expired(self) -> bool:
        if self._config is not None and time.monotonic() >= self._expires_at:
            self._config = None
        return self._config is None

    async def get(self) -> ReplyDelayConfig | None:
        return None if self._expired() else self._config

    async def set(self, config: ReplyDelayConfig, *, ttl_seconds: int) -> None:
        self._config = config
        self._expires_at = time.monotonic() + ttl_seconds

    async def ttl_seconds(self) -> int | None:
        if self._expired():
            return None
        return int(self._expires_at - time.monotonic())

    async def clear(self) -> None:
        self._config = None


def _encode_accounts(config: TestAccountConfig) -> str:
    return json.dumps(
        {
            "user_ids": sorted(config.user_ids),
            "mock_model": config.mock_model,
            "mock_opening": config.mock_opening,
        }
    )


def _decode_accounts(raw: str) -> TestAccountConfig:
    body = json.loads(raw)
    return TestAccountConfig(
        user_ids=frozenset(int(value) for value in body.get("user_ids", [])),
        # 키를 직접 넣을 때 빠뜨려도 비용이 드는 쪽으로 가지 않게 기본은 Mock이다.
        mock_model=bool(body.get("mock_model", True)),
        mock_opening=bool(body.get("mock_opening", False)),
    )


class RedisTestAccountStore:
    """부하 테스트 계정 설정을 Redis 키 하나에 둔다.

    SET needu:fault:test-accounts '{"user_ids":[1],"mock_model":false,"mock_opening":true}' EX 7200
    """

    def __init__(self, client: Redis, *, key: str = TEST_ACCOUNTS_KEY) -> None:
        self._client = client
        self._key = key

    async def get(self) -> TestAccountConfig | None:
        raw = await self._client.get(self._key)
        return None if raw is None else _decode_accounts(raw)

    async def set(self, config: TestAccountConfig, *, ttl_seconds: int) -> None:
        await self._client.set(self._key, _encode_accounts(config), ex=ttl_seconds)

    async def ttl_seconds(self) -> int | None:
        ttl = await self._client.ttl(self._key)
        return ttl if ttl >= 0 else None

    async def clear(self) -> None:
        await self._client.delete(self._key)


class InMemoryTestAccountStore:
    """Redis 없이 띄우는 로컬·테스트용. 프로세스 하나에서만 보인다."""

    def __init__(self) -> None:
        self._config: TestAccountConfig | None = None
        self._expires_at = 0.0

    def _current(self) -> TestAccountConfig | None:
        if self._config is not None and time.monotonic() >= self._expires_at:
            self._config = None
        return self._config

    async def get(self) -> TestAccountConfig | None:
        return self._current()

    async def set(self, config: TestAccountConfig, *, ttl_seconds: int) -> None:
        self._config = config
        self._expires_at = time.monotonic() + ttl_seconds

    async def ttl_seconds(self) -> int | None:
        if self._current() is None:
            return None
        return int(self._expires_at - time.monotonic())

    async def clear(self) -> None:
        self._config = None
