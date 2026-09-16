from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class CatalogRepository(Protocol):
    async def ping(self) -> bool: ...

    async def search(
        self,
        *,
        vector: Sequence[float],
        filters: Mapping[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]: ...
