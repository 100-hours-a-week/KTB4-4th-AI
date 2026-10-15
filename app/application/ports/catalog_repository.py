from collections.abc import Sequence
from typing import Protocol

from app.domain.recommendation import CatalogExclusion, CatalogSearchResult, VectorSpace


class CatalogRepository(Protocol):
    async def ping(self) -> bool: ...

    async def search(
        self,
        *,
        vector: Sequence[float],
        space: VectorSpace,
        embedding_space_id: str,
        limit: int,
        categories: Sequence[str] | None = None,
        gift_available_only: bool = False,
        exclusion: CatalogExclusion | None = None,
    ) -> Sequence[CatalogSearchResult]:
        """categories 가 있으면 그 카테고리 상품만 찾는다.

        gift_available_only 면 선물 목록에 나올 수 있는 상품만 찾는다.
        exclusion 의 대분류·세부 카테고리에 드는 상품은 찾지 않는다. 단어 조건은 호출한 쪽이
        결과의 상품명과 content 문서로 거른다.
        """
        ...
