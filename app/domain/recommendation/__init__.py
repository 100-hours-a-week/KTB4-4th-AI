"""V1 search-query policy, ranking, score floors, and diversity rules."""

from app.domain.recommendation.exclusions import CatalogExclusion, build_catalog_exclusion
from app.domain.recommendation.models import (
    CatalogProduct,
    CatalogSearchResult,
    ProductKey,
    RankedRecommendation,
    RecommendationGuide,
    RecommendationMode,
    RecommendationSignal,
    SearchQuery,
    VectorMatch,
    VectorSpace,
)
from app.domain.recommendation.queries import build_search_queries
from app.domain.recommendation.ranking import rank_recommendations

__all__ = [
    "CatalogExclusion",
    "CatalogProduct",
    "CatalogSearchResult",
    "ProductKey",
    "RankedRecommendation",
    "RecommendationGuide",
    "RecommendationMode",
    "RecommendationSignal",
    "SearchQuery",
    "VectorMatch",
    "VectorSpace",
    "build_catalog_exclusion",
    "build_search_queries",
    "rank_recommendations",
]
