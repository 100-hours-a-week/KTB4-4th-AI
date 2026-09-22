from __future__ import annotations

from collections.abc import Iterable

from app.domain.profile.models import TasteField
from app.domain.recommendation.models import (
    RecommendationMode,
    RecommendationSignal,
    SearchQuery,
    VectorSpace,
)

_FIELD_RULES: dict[
    TasteField,
    tuple[VectorSpace, frozenset[RecommendationMode]],
] = {
    TasteField.INTERESTS: (
        VectorSpace.CONTENT,
        frozenset({RecommendationMode.SELF, RecommendationMode.GIFT}),
    ),
    TasteField.HOBBIES: (
        VectorSpace.USAGE,
        frozenset({RecommendationMode.SELF, RecommendationMode.GIFT}),
    ),
    TasteField.WANTS: (
        VectorSpace.CONTENT,
        frozenset({RecommendationMode.SELF, RecommendationMode.GIFT}),
    ),
    TasteField.UNAFFORDABLE: (
        VectorSpace.GIFT,
        frozenset({RecommendationMode.GIFT}),
    ),
    TasteField.CONSUMABLES: (
        VectorSpace.CONTENT,
        frozenset({RecommendationMode.SELF}),
    ),
}


def _query_text(signal: RecommendationSignal, space: VectorSpace) -> str:
    if space == VectorSpace.USAGE:
        return f"{signal.value} 활동에서 사용하는 제품"
    if space == VectorSpace.GIFT:
        return f"선물하기 좋은 {signal.value} 관련 제품"
    if signal.field == TasteField.INTERESTS:
        return f"{signal.value} 관련 제품"
    return signal.value


def build_search_queries(
    signals: Iterable[RecommendationSignal],
    *,
    max_queries: int = 6,
) -> tuple[SearchQuery, ...]:
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")

    eligible = [
        signal for signal in signals if signal.field in _FIELD_RULES and signal.confidence > 0.0
    ]
    eligible.sort(
        key=lambda signal: (
            signal.confidence,
            signal.updated_at.timestamp(),
            signal.field.value,
            signal.value,
        ),
        reverse=True,
    )

    queries: list[SearchQuery] = []
    for index, signal in enumerate(eligible[:max_queries], start=1):
        space, modes = _FIELD_RULES[signal.field]
        queries.append(
            SearchQuery(
                query_id=f"q_{index}",
                signal=signal,
                modes=modes,
                space=space,
                text=_query_text(signal, space),
            )
        )
    return tuple(queries)
