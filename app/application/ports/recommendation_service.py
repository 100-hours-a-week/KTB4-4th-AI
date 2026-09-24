from collections.abc import Mapping
from typing import Any, Protocol


class RecommendationService(Protocol):
    async def recommend_lists(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
