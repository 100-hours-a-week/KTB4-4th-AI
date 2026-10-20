"""V1 search-query policy, ranking, score floors, and diversity rules."""

from app.domain.recommendation.exclusions import CatalogExclusion, build_catalog_exclusion
from app.domain.recommendation.intents import (
    INTENT_VERSION,
    InvalidSearchIntent,
    SearchIntent,
    intent_key,
    interest_field_name,
)
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
from app.domain.recommendation.queries import (
    build_explore_queries,
    build_search_queries,
    is_coverage_query,
    is_interest_query,
)
from app.domain.recommendation.ranking import rank_recommendations

__all__ = [
    "INTENT_VERSION",
    "CatalogExclusion",
    "CatalogProduct",
    "CatalogSearchResult",
    "InvalidSearchIntent",
    "ProductKey",
    "RankedRecommendation",
    "RecommendationGuide",
    "RecommendationMode",
    "RecommendationSignal",
    "SearchIntent",
    "SearchQuery",
    "VectorMatch",
    "VectorSpace",
    "build_catalog_exclusion",
    "build_explore_queries",
    "build_search_queries",
    "intent_key",
    "interest_field_name",
    "is_coverage_query",
    "is_interest_query",
    "rank_recommendations",
]
