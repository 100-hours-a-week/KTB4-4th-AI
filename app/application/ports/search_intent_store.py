from collections.abc import Mapping, Sequence
from typing import Protocol

from app.domain.recommendation import SearchIntent


class SearchIntentStore(Protocol):
    """intent_key 로 찾는 검색 의도 캐시. 사용자와 무관한 값이라 사용자끼리 같이 쓴다."""

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, SearchIntent]:
        """있는 키만 돌려준다. 저장된 값이 형식에 맞지 않으면 없는 것으로 본다."""
        ...

    async def save(self, key: str, intent: SearchIntent) -> None: ...
