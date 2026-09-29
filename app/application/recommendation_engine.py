from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.application.ports.catalog_repository import CatalogRepository
from app.application.ports.embedder import Embedder
from app.domain.recommendation import (
    CatalogSearchResult,
    ProductKey,
    RankedRecommendation,
    RecommendationGuide,
    RecommendationMode,
    RecommendationSignal,
    VectorMatch,
    VectorSpace,
    build_search_queries,
    rank_recommendations,
)

# 싫어하는 것·제약 조건과 이만큼 가까운 상품은 후보에서 뺀다. 코사인 유사도 기준이며
# 실제 카탈로그 분포를 보고 조정해야 하는 값이다.
EXCLUSION_SIMILARITY = 0.6
EXCLUSION_TOP_K = 40
# 선택 기준(preferences)과 이만큼 가까운 후보에는 점수를 얹는다.
PREFERENCE_SIMILARITY = 0.5
PREFERENCE_BONUS = 0.05
MAX_PREFERENCE_BONUS = 0.10


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


@dataclass(slots=True, frozen=True)
class RecommendationBatch:
    self: tuple[RankedRecommendation, ...]
    gift: tuple[RankedRecommendation, ...]


class RecommendationEngine:
    def __init__(
        self,
        *,
        embedder: Embedder,
        catalog: CatalogRepository,
        top_k_per_query: int = 80,
    ) -> None:
        if top_k_per_query <= 0:
            raise ValueError("top_k_per_query must be positive")
        self._embedder = embedder
        self._catalog = catalog
        self._top_k_per_query = top_k_per_query

    async def recommend(
        self,
        signals: Iterable[RecommendationSignal],
        *,
        now: datetime | None = None,
        limit: int = 20,
        guide: RecommendationGuide | None = None,
    ) -> RecommendationBatch:
        queries = build_search_queries(signals)
        if not queries:
            return RecommendationBatch(self=(), gift=())
        guide = guide or RecommendationGuide()

        # 쿼리, 제외, 선호 문장을 한 번에 임베딩한다.
        texts = [
            *(query.text for query in queries),
            *guide.exclusions,
            *guide.preferences,
        ]
        vectors = await self._embedder.embed_queries(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedder returned an unexpected number of query vectors")
        query_vectors = vectors[: len(queries)]
        exclusion_vectors = vectors[len(queries) : len(queries) + len(guide.exclusions)]
        preference_vectors = vectors[len(queries) + len(guide.exclusions) :]

        searches, exclusion_searches, preference_searches = await asyncio.gather(
            asyncio.gather(
                *(
                    self._search(vector, query.space, self._top_k_per_query)
                    for query, vector in zip(queries, query_vectors, strict=True)
                )
            ),
            asyncio.gather(
                *(
                    self._search(vector, VectorSpace.CONTENT, EXCLUSION_TOP_K)
                    for vector in exclusion_vectors
                )
            ),
            asyncio.gather(
                *(
                    self._search(vector, VectorSpace.CONTENT, self._top_k_per_query)
                    for vector in preference_vectors
                )
            ),
        )
        matches = [
            VectorMatch(
                query=query,
                product=result.product,
                similarity=result.similarity,
            )
            for query, results in zip(queries, searches, strict=True)
            for result in results
        ]
        excluded = _excluded_keys(matches, guide.exclusions, exclusion_searches)
        boosts = _preference_boosts(preference_searches)
        return RecommendationBatch(
            self=rank_recommendations(
                matches,
                mode=RecommendationMode.SELF,
                now=now,
                limit=limit,
                excluded=excluded,
                boosts=boosts,
            ),
            gift=rank_recommendations(
                matches,
                mode=RecommendationMode.GIFT,
                now=now,
                limit=limit,
                excluded=excluded,
                boosts=boosts,
            ),
        )

    async def _search(
        self,
        vector: Sequence[float],
        space: VectorSpace,
        limit: int,
    ) -> Sequence[CatalogSearchResult]:
        return await self._catalog.search(
            vector=vector,
            space=space,
            embedding_space_id=self._embedder.space_id,
            limit=limit,
        )


def _excluded_keys(
    matches: Iterable[VectorMatch],
    exclusions: Sequence[str],
    searches: Sequence[Sequence[CatalogSearchResult]],
) -> frozenset[ProductKey]:
    """싫어하는 것과 의미가 가깝거나 상품명에 그대로 들어간 후보를 고른다."""
    excluded = {
        result.product.key
        for results in searches
        for result in results
        if result.similarity >= EXCLUSION_SIMILARITY
    }
    terms = [_normalize(term) for term in exclusions if _normalize(term)]
    for match in matches:
        name = _normalize(match.product.name)
        if any(term in name for term in terms):
            excluded.add(match.product.key)
    return frozenset(excluded)


def _preference_boosts(
    searches: Sequence[Sequence[CatalogSearchResult]],
) -> Mapping[ProductKey, float]:
    boosts: dict[ProductKey, float] = {}
    for results in searches:
        for result in results:
            if result.similarity >= PREFERENCE_SIMILARITY:
                key = result.product.key
                boosts[key] = min(boosts.get(key, 0.0) + PREFERENCE_BONUS, MAX_PREFERENCE_BONUS)
    return boosts
