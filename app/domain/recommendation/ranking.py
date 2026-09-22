from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
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
    return match.query.signal.confidence * decay


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


def _reason(best: ScoredMatch, mode: RecommendationMode) -> str:
    signal = best.match.query.signal
    if mode == RecommendationMode.SELF:
        return f"{signal.value}에 대한 관심과 잘 맞는 상품이에요."
    if signal.field in {TasteField.INTERESTS, TasteField.HOBBIES} and signal.visibility in {
        Visibility.FRIENDS,
        Visibility.PUBLIC,
    }:
        return f"{signal.value}에 관심 있는 분에게 선물하기 좋은 상품이에요."
    return "평소 관심사와 잘 맞아 선물하기 좋은 상품이에요."


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
) -> tuple[RankedRecommendation, ...]:
    if not 1 <= limit <= DEFAULT_LIMIT:
        raise ValueError(f"limit must be between 1 and {DEFAULT_LIMIT}")
    threshold = DEFAULT_MIN_SCORE[mode] if min_score is None else min_score
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    evaluated_at = now or datetime.now(UTC)
    candidates: dict[ProductKey, list[VectorMatch]] = defaultdict(list)
    products = {}
    for match in matches:
        if mode not in match.query.modes:
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
        raw_score = scored_matches[0].contribution + bonus
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
            reason=_reason(scored_matches[0], mode),
            matches=scored_matches,
        )
        for rank, (key, score, scored_matches) in enumerate(selected)
    )
