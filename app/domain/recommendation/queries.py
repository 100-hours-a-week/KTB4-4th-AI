from __future__ import annotations

from collections.abc import Iterable

from app.domain.profile.models import PreferenceAspect, TasteField
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


# 대화에서 모으는 활성 취향 수(merger.MAX_ACTIVE_QUERY_SIGNALS)와 맞춘다.
MAX_QUERIES = 5
# 이 가중치 미만(희망·상상 답처럼 약한 신호)은 강한 신호가 모자랄 때만 쿼리로 쓴다.
MIN_PRIMARY_WEIGHT = 0.55
MIN_PRIMARY_QUERIES = 3
# 상황 취향(aspect=situation)으로 만드는 보조 usage 쿼리 수. 상위 취향만 쓴다.
MAX_SITUATION_QUERIES = 2


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
    max_queries: int = MAX_QUERIES,
) -> tuple[SearchQuery, ...]:
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")

    eligible = [
        signal for signal in signals if signal.field in _FIELD_RULES and signal.confidence > 0.0
    ]
    eligible.sort(
        key=lambda signal: (
            signal.weight,
            signal.confidence,
            signal.updated_at.timestamp(),
            signal.field.value,
            signal.value,
        ),
        reverse=True,
    )
    primary = [signal for signal in eligible if signal.weight >= MIN_PRIMARY_WEIGHT]
    if len(primary) < MIN_PRIMARY_QUERIES:
        # 강한 신호가 적으면 약한 신호로 채워서 추천이 비지 않게 한다.
        primary = eligible[: max(MIN_PRIMARY_QUERIES, len(primary))]

    queries: list[SearchQuery] = []
    for index, signal in enumerate(primary[:max_queries], start=1):
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
    queries.extend(_situation_queries(signals))
    return tuple(queries)


def _situation_queries(signals: Iterable[RecommendationSignal]) -> list[SearchQuery]:
    """상황 취향을 기존 usage 템플릿에 넣어 보조 쿼리를 만든다.

    상품 usage 문서가 활동·장소·상황으로 쓰여 있어서 상황 취향이 걸릴 수 있다.
    이미 적재한 문서와 틀을 맞추려고 템플릿은 hobbies와 같은 것을 쓴다.
    관심사 쿼리와 같은 상품에 함께 걸리면 다중 신호 보너스로 올라간다.
    """
    situations = sorted(
        (
            signal
            for signal in signals
            if signal.field == TasteField.PREFERENCES
            and signal.aspect == PreferenceAspect.SITUATION
            and signal.confidence > 0.0
        ),
        key=lambda signal: (signal.weight, signal.updated_at.timestamp(), signal.value),
        reverse=True,
    )
    return [
        SearchQuery(
            query_id=f"s_{index}",
            signal=signal,
            modes=frozenset({RecommendationMode.SELF, RecommendationMode.GIFT}),
            space=VectorSpace.USAGE,
            text=_query_text(signal, VectorSpace.USAGE),
        )
        for index, signal in enumerate(situations[:MAX_SITUATION_QUERIES], start=1)
    ]
