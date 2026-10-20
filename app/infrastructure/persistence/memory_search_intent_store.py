from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.domain.recommendation import SearchIntent


class InMemorySearchIntentStore:
    """Redis 가 없는 로컬 실행과 테스트용. 프로세스가 끝나면 사라진다."""

    def __init__(self) -> None:
        self._intents: dict[str, SearchIntent] = {}

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, SearchIntent]:
        return {key: self._intents[key] for key in keys if key in self._intents}

    async def save(self, key: str, intent: SearchIntent) -> None:
        self._intents[key] = intent
