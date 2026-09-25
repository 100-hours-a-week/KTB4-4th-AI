from collections.abc import Sequence
from typing import Protocol

from app.domain.recommendation import CatalogSearchResult, VectorSpace


class CatalogRepository(Protocol):
    async def ping(self) -> bool: ...

    async def search(
        self,
        *,
        vector: Sequence[float],
        space: VectorSpace,
        embedding_space_id: str,
        limit: int,
    ) -> Sequence[CatalogSearchResult]: ...
