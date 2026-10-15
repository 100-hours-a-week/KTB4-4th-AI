from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

from app.domain.profile.models import DeferralReason, TasteField, Visibility
from app.domain.recommendation.models import (
    ProductKey,
    RankedRecommendation,
    RecommendationMode,
    ScoredMatch,
    VectorMatch,
)

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


def _reason(
    best: ScoredMatch,
    mode: RecommendationMode,
    recipient_name: str | None = None,
) -> str:
    """추천 이유 문구. 선물 목록은 친구가 보므로 받는 사람 이름이 있으면 넣는다."""
    signal = best.match.query.signal
    owner = f"{recipient_name}님의 " if recipient_name else ""
    if signal.field == TasteField.PREFERENCES:
        # 취향 쿼리로 찾은 상품. 취향은 친구 공개 항목이라 선물 목록에서도 그대로 보여 준다.
        if mode == RecommendationMode.SELF:
            return f"{signal.value} 취향과 잘 맞는 상품이에요."
        return f"{owner}{signal.value} 취향에 맞춰 선물하기 좋은 상품이에요."
    if mode == RecommendationMode.SELF:
        return f"{signal.value}에 대한 관심과 잘 맞는 상품이에요."
    if signal.field in {TasteField.INTERESTS, TasteField.HOBBIES} and signal.visibility in {
        Visibility.FRIENDS,
        Visibility.PUBLIC,
    }:
        recipient = f"{recipient_name}님" if recipient_name else "분"
        return f"{signal.value}에 관심 있는 {recipient}에게 선물하기 좋은 상품이에요."
    return f"{owner}평소 관심사와 잘 맞아 선물하기 좋은 상품이에요."


def _select_with_query_diversity(
    ranked: list[tuple[ProductKey, float, tuple[ScoredMatch, ...]]],
    *,
    limit: int,
) -> list[tuple[ProductKey, float, tuple[ScoredMatch, ...]]]:
    selected: list[tuple[ProductKey, float, tuple[ScoredMatch, ...]]] = []
    selected_keys: set[ProductKey] = set()
    query_ids = {scored.match.query.query_id for _, _, matches in ranked for scored in matches}

    for query_id in sorted(query_ids):
        candidate = next(
            (
                item
                for item in ranked
                if item[0] not in selected_keys
                and any(match.match.query.query_id == query_id for match in item[2])
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_keys.add(candidate[0])
        if len(selected) == limit:
            break

    for candidate in ranked:
        if len(selected) == limit:
            break
        if candidate[0] in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(candidate[0])

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
    recipient_name: str | None = None,
) -> tuple[RankedRecommendation, ...]:
    """recipient_name 은 선물 목록 추천 이유에 넣는 받는 사람 이름이다. self 목록은 쓰지 않는다."""
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
    selected = _select_with_query_diversity(ranked, limit=limit)

    return tuple(
        RankedRecommendation(
            product=products[key],
            mode=mode,
            rank=rank,
            score=round(score, 6),
            reason=_reason(scored_matches[0], mode, recipient_name),
            matches=scored_matches,
        )
        for rank, (key, score, scored_matches) in enumerate(selected)
    )
