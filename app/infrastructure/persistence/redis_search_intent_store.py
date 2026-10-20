from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence

from redis.asyncio import Redis

from app.domain.recommendation import INTENT_VERSION, InvalidSearchIntent, SearchIntent

logger = logging.getLogger(__name__)

# 관심사 값과 분야로 정해지는 값이라 오래 둔다. 프롬프트를 바꾸면 INTENT_VERSION 이 키를 바꾼다.
DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 30


class RedisSearchIntentStore:
    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = "needu:search-intent:",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._client = client
        self._key_prefix = f"{key_prefix}{INTENT_VERSION}:"
        self._ttl_seconds = ttl_seconds

    def _key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, SearchIntent]:
        unique = list(dict.fromkeys(keys))
        if not unique:
            return {}
        raws = await self._client.mget([self._key(key) for key in unique])
        intents: dict[str, SearchIntent] = {}
        for key, raw in zip(unique, raws, strict=True):
            if raw is None:
                continue
            try:
                intents[key] = SearchIntent.from_payload(json.loads(raw))
            except (ValueError, TypeError, InvalidSearchIntent):
                logger.warning("stored search intent is invalid; ignoring it")
        return intents

    async def save(self, key: str, intent: SearchIntent) -> None:
        await self._client.set(
            self._key(key),
            json.dumps(intent.to_payload(), ensure_ascii=False, separators=(",", ":")),
            ex=self._ttl_seconds,
        )
