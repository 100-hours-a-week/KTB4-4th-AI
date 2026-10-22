from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from app.domain.profile.models import DeferralReason
from app.domain.recommendation.models import (
    ProductKey,
    RankedRecommendation,
    RecommendationMode,
    ScoredMatch,
    VectorMatch,
)
from app.domain.recommendation.reasons import recommendation_reason

DEFAULT_LIMIT = 20
DEFAULT_MIN_SCORE = {
    RecommendationMode.SELF: 0.15,
    RecommendationMode.GIFT: 0.10,
}
_HALF_LIFE_DAYS = {
    RecommendationMode.SELF: 14.0,
    RecommendationMode.GIFT: 180.0,
}
_DEFERRAL_WEIGHT = {
    None: 1.0,
    DeferralReason.TIMING: 1.15,
    DeferralReason.PRICE: 1.40,
    DeferralReason.JUSTIFICATION: 1.50,
}
_MULTI_SIGNAL_BONUS = 0.05
_MAX_MULTI_SIGNAL_BONUS = 0.10
_MAX_RAW_SCORE = {
    RecommendationMode.SELF: 1.0 + _MAX_MULTI_SIGNAL_BONUS,
    RecommendationMode.GIFT: 1.5 + _MAX_MULTI_SIGNAL_BONUS,
}


def _normalized_similarity(similarity: float) -> float:
    return min(max((similarity + 1.0) / 2.0, 0.0), 1.0)


def _decayed_confidence(match: VectorMatch, mode: RecommendationMode, now: datetime) -> float:
    age_days = max((now - match.query.signal.updated_at).total_seconds(), 0.0) / 86400.0
    decay = 0.5 ** (age_days / _HALF_LIFE_DAYS[mode])
    return match.query.signal.weight * decay


def _match_contribution(
    match: VectorMatch,
    mode: RecommendationMode,
    now: datetime,
) -> float:
    contribution = _normalized_similarity(match.similarity) * _decayed_confidence(
        match,
        mode,
        now,
    )
    if mode == RecommendationMode.GIFT:
        contribution *= _DEFERRAL_WEIGHT[match.query.signal.deferral_reason]
    return contribution


# 첫 바퀴로 쿼리마다 하나씩 고른 뒤에도 쿼리를 돌아가며 채운다. 상품이 많은 관심사는 검색 의도
# 쿼리의 유사도가 높게 나와서, 점수순으로만 채우면 한 관심사가 목록을 독차지한다.
# 1등보다 이만큼 넘게 낮은 후보는 돌림에서 빼서, 맞는 상품이 없는 쿼리가 자리를 나눠 갖지 않게 한다.
# 돌림으로 못 채운 자리는 점수순으로 채운다.
ROUND_ROBIN_SCORE_GAP = 0.08
# 탐색 추천(카탈로그에 상품이 없는 관심사 대신 찾은 상품)의 자리. 관심사 하나당, 목록 하나당 상한을
# 둬서 탐색이 목록을 덮지 않게 한다. 관심사에 맞는 상품이 모자라 목록이 이보다 짧아지면
# 탐색 상품으로 이만큼까지는 채운다.
EXPLORE_PER_SIGNAL = 2
MAX_EXPLORE = 4
MIN_LIST_SIZE = 10

_Ranked = tuple[ProductKey, float, tuple[ScoredMatch, ...]]


def _is_explore(item: _Ranked) -> bool:
    """탐색 쿼리로만 찾은 상품. 관심사 쿼리에도 걸렸으면 관심사 상품으로 본다."""
    return all(scored.match.query.exploration for scored in item[2])


def _pick_explore(
    ranked: list[_Ranked],
    *,
    total: int,
    per_signal: int | None,
    taken: set[ProductKey],
) -> list[_Ranked]:
    """탐색 상품을 관심사마다 돌아가며 고른다. ranked 는 점수순이다."""
    groups: dict[str, list[_Ranked]] = {}
    for item in ranked:
        if item[0] not in taken:
            groups.setdefault(item[2][0].match.query.query_id, []).append(item)
    picked: list[_Ranked] = []
    depth = 0
    while len(picked) < total and (per_signal is None or depth < per_signal):
        added = False
        for members in groups.values():
            if len(picked) == total:
                break
            if depth < len(members):
                picked.append(members[depth])
                added = True
        if not added:
            break
        depth += 1
    taken.update(item[0] for item in picked)
    return picked


def _select_with_query_diversity(
    ranked: list[tuple[ProductKey, float, tuple[ScoredMatch, ...]]],
    *,
    limit: int,
) -> list[tuple[ProductKey, float, tuple[ScoredMatch, ...]]]:
    selected: list[tuple[ProductKey, float, tuple[ScoredMatch, ...]]] = []
    selected_keys: set[ProductKey] = set()
    query_ids = sorted(
        {scored.match.query.query_id for _, _, matches in ranked for scored in matches}
    )
    # ranked 가 점수순이라 쿼리별 후보도 점수순이다.
    by_query = {
        query_id: [
            item
            for item in ranked
            if any(match.match.query.query_id == query_id for match in item[2])
        ]
        for query_id in query_ids
    }

    def take(candidate: tuple[ProductKey, float, tuple[ScoredMatch, ...]]) -> None:
        selected.append(candidate)
        selected_keys.add(candidate[0])

    def next_for(query_id: str, floor: float):
        return next(
            (
                item
                for item in by_query[query_id]
                if item[0] not in selected_keys and item[1] >= floor
            ),
            None,
        )

    for query_id in query_ids:
        if len(selected) == limit:
            break
        candidate = next_for(query_id, float("-inf"))
        if candidate is not None:
            take(candidate)

    floor = (ranked[0][1] - ROUND_ROBIN_SCORE_GAP) if ranked else 0.0
    progressed = True
    while len(selected) < limit and progressed:
        progressed = False
        for query_id in query_ids:
            if len(selected) == limit:
                break
            candidate = next_for(query_id, floor)
            if candidate is not None:
                take(candidate)
                progressed = True

    for candidate in ranked:
        if len(selected) == limit:
            break
        if candidate[0] not in selected_keys:
            take(candidate)

    selected.sort(key=lambda item: (-item[1], item[0].platform, item[0].external_id))
    return selected


def rank_recommendations(
    matches: Iterable[VectorMatch],
    *,
    mode: RecommendationMode,
    now: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
    min_score: float | None = None,
    excluded: frozenset[ProductKey] = frozenset(),
    boosts: Mapping[ProductKey, float] | None = None,
    boost_labels: Mapping[ProductKey, str] | None = None,
    recipient_name: str | None = None,
) -> tuple[RankedRecommendation, ...]:
    """recipient_name 은 선물 목록 추천 이유에 넣는 받는 사람 이름이다. self 목록은 쓰지 않는다.

    boost_labels 는 취향 가산을 준 선호 문장이다. 추천 이유에 "~ 취향도 반영했어요"로 붙는다.
    탐색 상품은 관심사 상품 뒤에 두고, 점수도 관심사 상품의 마지막 점수를 넘지 않게 맞춘다.
    """
    if not 1 <= limit <= DEFAULT_LIMIT:
        raise ValueError(f"limit must be between 1 and {DEFAULT_LIMIT}")
    threshold = DEFAULT_MIN_SCORE[mode] if min_score is None else min_score
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    evaluated_at = now or datetime.now(UTC)
    candidates: dict[ProductKey, list[VectorMatch]] = defaultdict(list)
    products = {}
    for match in matches:
        if mode not in match.query.modes or match.product.key in excluded:
            continue
        # 선물 목록은 gift_available 상품만 보여 준다. content·usage 쿼리는 두 모드가 같이 쓰므로
        # gift_available 이 아닌 상품도 검색 결과에 섞여 들어온다.
        if mode is RecommendationMode.GIFT and not match.product.gift_available:
            continue
        candidates[match.product.key].append(match)
        products[match.product.key] = match.product

    ranked: list[tuple[ProductKey, float, tuple[ScoredMatch, ...]]] = []
    for key, candidate_matches in candidates.items():
        scored_matches = tuple(
            sorted(
                (
                    ScoredMatch(
                        match=match,
                        contribution=_match_contribution(match, mode, evaluated_at),
                    )
                    for match in candidate_matches
                ),
                key=lambda scored: scored.contribution,
                reverse=True,
            )
        )
        unique_signal_count = len({scored.match.query.query_id for scored in scored_matches})
        bonus = min(max(unique_signal_count - 1, 0) * _MULTI_SIGNAL_BONUS, 0.10)
        raw_score = scored_matches[0].contribution + bonus + (boosts or {}).get(key, 0.0)
        score = min(max(raw_score / _MAX_RAW_SCORE[mode], 0.0), 1.0)
        if score >= threshold:
            ranked.append((key, score, scored_matches))

    ranked.sort(key=lambda item: (-item[1], item[0].platform, item[0].external_id))
    matched = [item for item in ranked if not _is_explore(item)]
    explore_pool = [item for item in ranked if _is_explore(item)]
    taken: set[ProductKey] = set()
    explore = _pick_explore(
        explore_pool, total=min(MAX_EXPLORE, limit), per_signal=EXPLORE_PER_SIGNAL, taken=taken
    )
    selected = _select_with_query_diversity(matched, limit=limit - len(explore))
    shortfall = min(MIN_LIST_SIZE, limit) - len(selected) - len(explore)
    if shortfall > 0:
        explore += _pick_explore(explore_pool, total=shortfall, per_signal=None, taken=taken)
    explore.sort(key=lambda item: (-item[1], item[0].platform, item[0].external_id))
    ceiling = selected[-1][1] if selected else None
    final = [
        *selected,
        *(
            (key, min(score, ceiling) if ceiling is not None else score, scored)
            for key, score, scored in explore
        ),
    ]

    return tuple(
        RankedRecommendation(
            product=products[key],
            mode=mode,
            rank=rank,
            score=round(score, 6),
            reason=recommendation_reason(
                scored_matches,
                mode,
                rank=rank,
                recipient_name=recipient_name,
                boost_label=(boost_labels or {}).get(key),
            ),
            matches=scored_matches,
            boost_label=(boost_labels or {}).get(key),
        )
        for rank, (key, score, scored_matches) in enumerate(final)
    )
