from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.domain.profile.models import PreferenceAspect, TasteField
from app.domain.recommendation.intents import SearchIntent, intent_key
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


# 관심사 쿼리 수. 노출 한도와 상관없이 저장된 관심사 전체에서 분야를 돌아가며 고른다.
MAX_QUERIES = 8
# 이 가중치 미만(희망·상상 답처럼 약한 신호)은 강한 신호가 모자랄 때만 쿼리로 쓴다.
MIN_PRIMARY_WEIGHT = 0.55
MIN_PRIMARY_QUERIES = 3
# 취향을 관심사와 조합한 content 쿼리 수.
MAX_TASTE_QUERIES = 6
# 상황 취향(aspect=situation)으로 만드는 보조 usage 쿼리 수.
MAX_SITUATION_QUERIES = 3
# 대상이 없는 취향("미니멀한 디자인", "사람 없는 곳")을 짝지어 볼 상위 관심사 수.
MAX_CROSS_INTERESTS = 2
# 상품 content 문서와 맞는 취향 측면. 동기는 상품 문서와 맞지 않아 쿼리로 쓰지 않는다.
_CONTENT_TASTE_ASPECTS = frozenset(
    {None, PreferenceAspect.ATTRIBUTE, PreferenceAspect.SENSORY, PreferenceAspect.CRITERION}
)
_BOTH_MODES = frozenset({RecommendationMode.SELF, RecommendationMode.GIFT})
_GIFT_ONLY = frozenset({RecommendationMode.GIFT})
# 선물 목록용 gift 공간 쿼리를 더하는 필드. 관심사·취미·갖고 싶은 것은 content·usage 쿼리를
# 두 모드가 같이 쓰지만, gift_available 상품의 gift 문서와도 맞춰 봐야 선물 목록이 채워진다.
_GIFT_QUERY_FIELDS = frozenset({TasteField.INTERESTS, TasteField.HOBBIES, TasteField.WANTS})


def _query_text(
    signal: RecommendationSignal,
    space: VectorSpace,
    intents: Mapping[str, SearchIntent] | None = None,
) -> str:
    """검색 의도가 있으면 그 공간의 의도 문장을, 없으면 템플릿 문장을 쓴다."""
    intent = (intents or {}).get(intent_key(signal.value, signal.taxonomy_path))
    if intent is not None:
        return intent.text(space)
    if space == VectorSpace.USAGE:
        return f"{signal.value} 활동에서 사용하는 제품"
    if space == VectorSpace.GIFT:
        # 상품 gift 문서가 받는 사람을 취향으로 쓰므로 같은 문형으로 묻는다.
        return f"{signal.value}에 관심 있는 사람이 받으면 좋은 선물"
    if signal.field == TasteField.INTERESTS:
        return f"{signal.value} 관련 제품"
    return signal.value


def _by_weight(signals: Iterable[RecommendationSignal]) -> list[RecommendationSignal]:
    return sorted(
        signals,
        key=lambda signal: (
            signal.weight,
            signal.confidence,
            signal.updated_at.timestamp(),
            signal.field.value,
            signal.value,
        ),
        reverse=True,
    )


def _spread(signals: list[RecommendationSignal]) -> list[RecommendationSignal]:
    """가중치 순서를 지키되 같은 분야·축이 앞자리를 몰아 차지하지 않게 분류마다 돌아가며 둔다."""
    groups: dict[object, list[RecommendationSignal]] = {}
    for index, signal in enumerate(signals):
        groups.setdefault(signal.taxonomy_path or index, []).append(signal)
    ordered: list[RecommendationSignal] = []
    depth = 0
    while len(ordered) < len(signals):
        for members in groups.values():
            if depth < len(members):
                ordered.append(members[depth])
        depth += 1
    return ordered


def _with_target(signal: RecommendationSignal) -> str | None:
    target = (signal.target or "").strip()
    if not target:
        return None
    # 한국어는 꾸미는 말이 앞에 온다. "자극적" + 영화 → "자극적 영화"
    return signal.value if target in signal.value else f"{signal.value} {target}"


def build_search_queries(
    signals: Iterable[RecommendationSignal],
    *,
    max_queries: int = MAX_QUERIES,
    intents: Mapping[str, SearchIntent] | None = None,
) -> tuple[SearchQuery, ...]:
    """저장된 취향·관심사로 검색 쿼리를 만든다.

    관심사마다 쿼리를 하나씩 만들고, 취향은 붙은 관심사나 상위 관심사와 조합해 쿼리를 더한다.
    쿼리가 다양할수록 여러 종류의 상품이 후보에 오르고,
    랭킹이 쿼리마다 하나씩 먼저 뽑아 목록을 채운다.

    intents 는 intent_key 로 찾는 관심사별 검색 의도다. 의도가 있는 관심사의 쿼리는 의도 문장을
    쓰고, 없으면 템플릿 문장을 쓴다. 취향 쿼리는 사용자마다 달라서 의도를 쓰지 않는다.
    """
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    signals = list(signals)

    eligible = _by_weight(
        signal for signal in signals if signal.field in _FIELD_RULES and signal.confidence > 0.0
    )
    primary = [signal for signal in eligible if signal.weight >= MIN_PRIMARY_WEIGHT]
    if len(primary) < MIN_PRIMARY_QUERIES:
        # 강한 신호가 적으면 약한 신호로 채워서 추천이 비지 않게 한다.
        primary = eligible[: max(MIN_PRIMARY_QUERIES, len(primary))]
    interests = _spread(primary)[:max_queries]

    queries: list[SearchQuery] = []
    for index, signal in enumerate(interests, start=1):
        space, modes = _FIELD_RULES[signal.field]
        queries.append(
            SearchQuery(
                query_id=f"q_{index}",
                signal=signal,
                modes=modes,
                space=space,
                text=_query_text(signal, space, intents),
            )
        )
    queries.extend(
        SearchQuery(
            query_id=f"g_{index}",
            signal=signal,
            modes=_GIFT_ONLY,
            space=VectorSpace.GIFT,
            text=_query_text(signal, VectorSpace.GIFT, intents),
        )
        for index, signal in enumerate(
            (signal for signal in interests if signal.field in _GIFT_QUERY_FIELDS),
            start=1,
        )
    )
    anchors = [
        signal for signal in interests if signal.field in {TasteField.INTERESTS, TasteField.HOBBIES}
    ][:MAX_CROSS_INTERESTS]
    queries.extend(_taste_queries(signals, anchors))
    queries.extend(_situation_queries(signals, anchors))
    return tuple(_unique_texts(queries))


def _tastes(
    signals: Iterable[RecommendationSignal],
    aspects: frozenset[PreferenceAspect | None],
) -> list[RecommendationSignal]:
    return _spread(
        _by_weight(
            signal
            for signal in signals
            if signal.field == TasteField.PREFERENCES
            and signal.aspect in aspects
            and signal.confidence > 0.0
        )
    )


def _taste_queries(
    signals: Iterable[RecommendationSignal],
    anchors: list[RecommendationSignal],
) -> list[SearchQuery]:
    """취향을 관심사와 조합한 content 쿼리.

    관심사가 붙은 취향("고소한 커피", "무채색 캠핑 장비")은 그대로 쓰고,
    대상 없는 취향("미니멀한 디자인")은 상위 관심사마다 짝지어
    "캠핑 관련 제품 중 미니멀한 디자인"처럼 쓴다.
    """
    texts: list[tuple[RecommendationSignal, str, RecommendationSignal | None]] = []
    for signal in _tastes(signals, _CONTENT_TASTE_ASPECTS):
        base = _with_target(signal)
        if base is not None:
            texts.append((signal, f"{base} 관련 제품", None))
        else:
            texts.extend(
                (signal, f"{anchor.value} 관련 제품 중 {signal.value}", anchor)
                for anchor in anchors
            )
    return [
        SearchQuery(
            query_id=f"t_{index}",
            signal=signal,
            modes=_BOTH_MODES,
            space=VectorSpace.CONTENT,
            text=text,
            anchor=anchor,
        )
        for index, (signal, text, anchor) in enumerate(texts[:MAX_TASTE_QUERIES], start=1)
    ]


def _situation_queries(
    signals: Iterable[RecommendationSignal],
    anchors: list[RecommendationSignal],
) -> list[SearchQuery]:
    """상황 취향을 기존 usage 템플릿에 넣어 보조 쿼리를 만든다.

    상품 usage 문서가 활동·장소·상황으로 쓰여 있어서 상황 취향이 걸릴 수 있다.
    대상 없는 상황 취향("사람 없는 곳")은 가장 강한 관심사와 합쳐 "사람 없는 곳 캠핑"처럼 쓴다.
    관심사 쿼리와 같은 상품에 함께 걸리면 다중 신호 보너스로 올라간다.
    """
    queries: list[SearchQuery] = []
    for signal in _tastes(signals, frozenset({PreferenceAspect.SITUATION})):
        base = _with_target(signal)
        anchor = None
        if base is None:
            if anchors and anchors[0].value not in signal.value:
                anchor = anchors[0]
                base = f"{signal.value} {anchor.value}"
            else:
                base = signal.value
        queries.append(
            SearchQuery(
                query_id=f"s_{len(queries) + 1}",
                signal=signal,
                modes=_BOTH_MODES,
                space=VectorSpace.USAGE,
                text=f"{base} 활동에서 사용하는 제품",
                anchor=anchor,
            )
        )
        if len(queries) == MAX_SITUATION_QUERIES:
            break
    return queries


def _unique_texts(queries: Iterable[SearchQuery]) -> list[SearchQuery]:
    seen: set[tuple[VectorSpace, str]] = set()
    unique: list[SearchQuery] = []
    for query in queries:
        key = (query.space, query.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
    return unique


def is_interest_query(query: SearchQuery) -> bool:
    """관심사 하나를 찾는 쿼리(q_, g_). 취향·상황 쿼리와 탐색 쿼리는 아니다."""
    return not query.exploration and query.signal.field != TasteField.PREFERENCES


def is_coverage_query(query: SearchQuery) -> bool:
    """카탈로그에 그 관심사의 상품이 있는지 보는 쿼리.

    gift 공간은 유사도가 전반적으로 높아서 판정에 쓰지 않는다.
    """
    return is_interest_query(query) and query.space != VectorSpace.GIFT


def build_explore_queries(
    queries: Iterable[SearchQuery],
    *,
    intents: Mapping[str, SearchIntent] | None,
) -> tuple[SearchQuery, ...]:
    """검색 의도가 있는 관심사마다 탐색 쿼리를 하나씩 만든다.

    쓸지는 1단계 검색 뒤에 엔진이 정한다. 상품 content 문서와 같은 문형이라 content 공간에서 찾는다.
    사용자 취향은 붙이지 않는다. "미니멀한 디자인"을 붙이면 "미니" 글자에 걸린 상품이 올라왔다.
    """
    if not intents:
        return ()
    explore: list[SearchQuery] = []
    seen: set[str] = set()
    for query in queries:
        if not is_coverage_query(query):
            continue
        key = intent_key(query.signal.value, query.signal.taxonomy_path)
        intent = intents.get(key)
        if intent is None or key in seen:
            continue
        seen.add(key)
        explore.append(
            SearchQuery(
                query_id=f"e_{len(explore) + 1}",
                signal=query.signal,
                modes=_BOTH_MODES,
                space=VectorSpace.CONTENT,
                text=intent.explore,
                exploration=True,
            )
        )
    return tuple(explore)
