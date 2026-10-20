from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.application.ports.catalog_repository import CatalogRepository
from app.application.ports.embedder import Embedder
from app.domain.recommendation import (
    CatalogExclusion,
    CatalogSearchResult,
    ProductKey,
    RankedRecommendation,
    RecommendationGuide,
    RecommendationMode,
    RecommendationSignal,
    SearchIntent,
    SearchQuery,
    VectorMatch,
    VectorSpace,
    build_explore_queries,
    build_search_queries,
    intent_key,
    is_coverage_query,
    is_interest_query,
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
# 검색 의도 쿼리의 1위가 이보다 낮으면 카탈로그에 그 관심사의 상품이 없는 것으로 본다.
# 그 관심사로 찾은 상품은 버리고 탐색 쿼리로 찾은 상품을 대신 넣는다.
# 의도를 세 번씩 만들어 잰 1위 유사도는 이랬다. 평가셋으로 다시 맞춰야 하는 값이다.
#   상품 있음: 커피·차·독서·콘솔게임·캠핑 0.67~0.76, 와인(와인잔) 0.63, 산책(운동화·백팩) 0.64~0.65
#   상품 없음: 볼링(골프 장갑) 0.61~0.62, 스키 0.57~0.60, 뜨개질 0.56~0.59
# 템플릿 쿼리는 유사도 분포가 달라 판정하지 않는다.
COVERAGE_MIN_SIMILARITY = 0.62
EXPLORE_TOP_K = 40


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
        categories: Sequence[str] | None = None,
        recipient_name: str | None = None,
        intents: Mapping[str, SearchIntent] | None = None,
    ) -> RecommendationBatch:
        """categories 를 주면 그 카테고리 안에서만 찾는다(v2 카테고리 탭).

        recipient_name 은 선물 목록 추천 이유에 넣는 받는 사람 이름이다.
        intents 는 관심사별 검색 의도다. 없는 관심사는 템플릿 쿼리로 찾는다.
        """
        queries = build_search_queries(signals, intents=intents)
        if not queries:
            return RecommendationBatch(self=(), gift=())
        guide = guide or RecommendationGuide()
        explore_queries = build_explore_queries(queries, intents=intents)

        # 쿼리, 탐색, 제외, 선호 문장을 한 번에 임베딩한다. 탐색 쿼리는 쓸지 모르지만
        # 같이 넣어 두면 2단계 검색을 위해 임베딩을 다시 부르지 않는다.
        texts = [
            *(query.text for query in queries),
            *(query.text for query in explore_queries),
            *guide.exclusions,
            *guide.preferences,
        ]
        vectors = await self._embedder.embed_queries(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedder returned an unexpected number of query vectors")
        cursor = 0
        query_vectors = vectors[cursor : cursor + len(queries)]
        cursor += len(queries)
        explore_vectors = vectors[cursor : cursor + len(explore_queries)]
        cursor += len(explore_queries)
        exclusion_vectors = vectors[cursor : cursor + len(guide.exclusions)]
        cursor += len(guide.exclusions)
        preference_vectors = vectors[cursor:]

        searches, exclusion_searches, preference_searches = await asyncio.gather(
            asyncio.gather(
                *(
                    self._search(
                        vector,
                        query.space,
                        self._top_k_per_query,
                        categories=categories,
                        # 선물 전용 쿼리는 처음부터 gift_available 상품 안에서 찾아야
                        # top_k 를 선물 목록 밖 상품에 쓰지 않는다. gift 문서는 gift_available
                        # 상품에만 만들지만, false 로 바뀐 상품의 예전 문서가 남아 있을 수 있다.
                        gift_available_only=query.modes == {RecommendationMode.GIFT},
                        exclusion=guide.catalog_exclusion,
                    )
                    for query, vector in zip(queries, query_vectors, strict=True)
                )
            ),
            asyncio.gather(
                *(
                    self._search(
                        vector, VectorSpace.CONTENT, EXCLUSION_TOP_K, categories=categories
                    )
                    for vector in exclusion_vectors
                )
            ),
            asyncio.gather(
                *(
                    self._search(
                        vector,
                        VectorSpace.CONTENT,
                        self._top_k_per_query,
                        categories=categories,
                    )
                    for vector in preference_vectors
                )
            ),
        )

        # 2단계: 카탈로그에 상품이 없는 관심사는 1단계 결과를 버리고 탐색 쿼리로 다시 찾는다.
        uncovered = _uncovered(queries, searches, intents)
        explore = [
            (query, vector)
            for query, vector in zip(explore_queries, explore_vectors, strict=True)
            if _signal_key(query) in uncovered
        ]
        explore_searches = await asyncio.gather(
            *(
                self._search(
                    vector,
                    query.space,
                    EXPLORE_TOP_K,
                    categories=categories,
                    exclusion=guide.catalog_exclusion,
                )
                for query, vector in explore
            )
        )
        matches = [
            VectorMatch(
                query=query,
                product=result.product,
                similarity=result.similarity,
            )
            for query, results in (
                *zip(queries, searches, strict=True),
                *zip((query for query, _ in explore), explore_searches, strict=True),
            )
            if not _drops_with(query, uncovered)
            for result in results
        ]
        excluded = _excluded_keys(
            matches, guide.exclusions, exclusion_searches, guide.catalog_exclusion
        )
        boosts, boost_labels = _preference_boosts(guide.preferences, preference_searches)
        return RecommendationBatch(
            self=rank_recommendations(
                matches,
                mode=RecommendationMode.SELF,
                now=now,
                limit=limit,
                excluded=excluded,
                boosts=boosts,
                boost_labels=boost_labels,
            ),
            gift=rank_recommendations(
                matches,
                mode=RecommendationMode.GIFT,
                now=now,
                limit=limit,
                excluded=excluded,
                boosts=boosts,
                boost_labels=boost_labels,
                recipient_name=recipient_name,
            ),
        )

    async def _search(
        self,
        vector: Sequence[float],
        space: VectorSpace,
        limit: int,
        *,
        categories: Sequence[str] | None = None,
        gift_available_only: bool = False,
        exclusion: CatalogExclusion | None = None,
    ) -> Sequence[CatalogSearchResult]:
        return await self._catalog.search(
            vector=vector,
            space=space,
            embedding_space_id=self._embedder.space_id,
            limit=limit,
            categories=categories,
            gift_available_only=gift_available_only,
            exclusion=exclusion,
        )


def _excluded_keys(
    matches: Iterable[VectorMatch],
    exclusions: Sequence[str],
    searches: Sequence[Sequence[CatalogSearchResult]],
    catalog_exclusion: CatalogExclusion,
) -> frozenset[ProductKey]:
    """싫어하는 것과 의미가 가깝거나 상품명에 그대로 들어간 후보를 고른다.

    온보딩 제외 조건은 검색이 대분류·세부 카테고리로 이미 걸렀고, 여기서 단어 조건을 본다.
    """
    excluded = {
        result.product.key
        for results in searches
        for result in results
        if result.similarity >= EXCLUSION_SIMILARITY
    }
    terms = [_normalize(term) for term in exclusions if _normalize(term)]
    for match in matches:
        name = _normalize(match.product.name)
        if any(term in name for term in terms) or catalog_exclusion.blocks(match.product):
            excluded.add(match.product.key)
    return frozenset(excluded)


def _signal_key(query: SearchQuery) -> str:
    return intent_key(query.signal.value, query.signal.taxonomy_path)


def _drops_with(query: SearchQuery, uncovered: frozenset[str]) -> bool:
    """상품이 없다고 판정된 관심사의 쿼리, 그 관심사와 짝지은 취향·상황 쿼리."""
    if is_interest_query(query) and _signal_key(query) in uncovered:
        return True
    anchor = query.anchor
    return anchor is not None and intent_key(anchor.value, anchor.taxonomy_path) in uncovered


def _uncovered(
    queries: Sequence[SearchQuery],
    searches: Sequence[Sequence[CatalogSearchResult]],
    intents: Mapping[str, SearchIntent] | None,
) -> frozenset[str]:
    """검색 의도로 찾았는데 1위도 COVERAGE_MIN_SIMILARITY 보다 낮은 관심사의 키."""
    if not intents:
        return frozenset()
    best: dict[str, float] = {}
    for query, results in zip(queries, searches, strict=True):
        key = _signal_key(query)
        if not is_coverage_query(query) or key not in intents:
            continue
        top = max((result.similarity for result in results), default=0.0)
        best[key] = max(best.get(key, 0.0), top)
    return frozenset(key for key, top in best.items() if top < COVERAGE_MIN_SIMILARITY)


def _preference_boosts(
    preferences: Sequence[str],
    searches: Sequence[Sequence[CatalogSearchResult]],
) -> tuple[Mapping[ProductKey, float], Mapping[ProductKey, str]]:
    """선호 문장과 가까운 상품의 가산점과, 추천 이유에 쓸 가장 가까운 선호 문장."""
    boosts: dict[ProductKey, float] = {}
    nearest: dict[ProductKey, tuple[float, str]] = {}
    for preference, results in zip(preferences, searches, strict=True):
        for result in results:
            if result.similarity >= PREFERENCE_SIMILARITY:
                key = result.product.key
                boosts[key] = min(boosts.get(key, 0.0) + PREFERENCE_BONUS, MAX_PREFERENCE_BONUS)
                if result.similarity > nearest.get(key, (0.0, ""))[0]:
                    nearest[key] = (result.similarity, preference)
    return boosts, {key: label for key, (_, label) in nearest.items()}
