from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from app.application.ports.catalog_repository import CatalogRepository
from app.application.ports.embedder import Embedder
from app.domain.recommendation import (
    RankedRecommendation,
    RecommendationMode,
    RecommendationSignal,
    VectorMatch,
    build_search_queries,
    rank_recommendations,
)


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
    ) -> RecommendationBatch:
        queries = build_search_queries(signals)
        if not queries:
            return RecommendationBatch(self=(), gift=())

        vectors = await self._embedder.embed_queries([query.text for query in queries])
        if len(vectors) != len(queries):
            raise ValueError("embedder returned an unexpected number of query vectors")

        searches = await asyncio.gather(
            *(
                self._catalog.search(
                    vector=vector,
                    space=query.space,
                    embedding_space_id=self._embedder.space_id,
                    limit=self._top_k_per_query,
                )
                for query, vector in zip(queries, vectors, strict=True)
            )
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
        return RecommendationBatch(
            self=rank_recommendations(
                matches,
                mode=RecommendationMode.SELF,
                now=now,
                limit=limit,
            ),
            gift=rank_recommendations(
                matches,
                mode=RecommendationMode.GIFT,
                now=now,
                limit=limit,
            ),
        )
