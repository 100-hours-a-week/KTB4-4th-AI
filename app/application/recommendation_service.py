from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.application.recommendation_engine import RecommendationBatch, RecommendationEngine
from app.domain.profile.models import DeferralReason, TasteField, Visibility
from app.domain.recommendation import (
    RankedRecommendation,
    RecommendationMode,
    RecommendationSignal,
    build_search_queries,
)

_PROFILE_FIELDS = (
    TasteField.INTERESTS,
    TasteField.HOBBIES,
    TasteField.WANTS,
    TasteField.UNAFFORDABLE,
    TasteField.CONSUMABLES,
)


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("profile updatedAt must be a datetime")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _recommendation_signals(payload: Mapping[str, Any]) -> tuple[RecommendationSignal, ...]:
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be an object")

    signals: list[RecommendationSignal] = []
    for field in _PROFILE_FIELDS:
        items = profile.get(field.value, [])
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ValueError(f"profile {field.value} must be an array")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"profile {field.value} items must be objects")
            reason = item.get("deferralReason")
            signals.append(
                RecommendationSignal(
                    field=field,
                    value=str(item["value"]),
                    confidence=float(item["confidence"]),
                    visibility=Visibility(str(item["visibility"])),
                    updated_at=_parse_datetime(item["updatedAt"]),
                    deferral_reason=DeferralReason(str(reason)) if reason else None,
                )
            )
    return tuple(signals)


def _items(recommendations: Sequence[RankedRecommendation]) -> list[dict[str, str]]:
    return [
        {
            "platform": recommendation.product.key.platform,
            "externalId": recommendation.product.key.external_id,
            "reason": recommendation.reason,
        }
        for recommendation in recommendations
    ]


def _result(
    recommendations: Sequence[RankedRecommendation],
    *,
    mode: RecommendationMode,
    has_signal: bool,
) -> dict[str, Any]:
    returned = len(recommendations)
    empty_reason = None
    if returned == 0:
        empty_reason = "NO_RELEVANT_CANDIDATE" if has_signal else "NO_PROFILE_SIGNAL"
    return {
        "recommendationId": f"r_{mode.value}_{uuid4().hex[:12]}",
        "generatedAt": datetime.now(UTC).isoformat(),
        "mode": mode.value,
        "priceRange": None,
        "items": _items(recommendations),
        "emptyReason": empty_reason,
        "suggestion": "MORE_CONVERSATION" if empty_reason else None,
        "funnel": {
            "retrieved": returned,
            "afterHardFilter": returned,
            "afterScoreFloor": returned,
            "returned": returned,
        },
    }


class V1RecommendationService:
    def __init__(self, engine: RecommendationEngine) -> None:
        self._engine = engine

    async def recommend_lists(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        signals = _recommendation_signals(payload)
        queries = build_search_queries(signals)
        batch = await self._engine.recommend(signals)
        return {
            "self": _result(
                batch.self,
                mode=RecommendationMode.SELF,
                has_signal=any(RecommendationMode.SELF in query.modes for query in queries),
            ),
            "gift": _result(
                batch.gift,
                mode=RecommendationMode.GIFT,
                has_signal=any(RecommendationMode.GIFT in query.modes for query in queries),
            ),
        }

    async def recommend(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        signals = _recommendation_signals(payload)
        queries = build_search_queries(signals)
        mode = RecommendationMode(str(payload["mode"]))
        limit = int(payload.get("limit", 20))
        batch: RecommendationBatch = await self._engine.recommend(signals, limit=limit)
        recommendations = batch.self if mode == RecommendationMode.SELF else batch.gift
        return _result(
            recommendations,
            mode=mode,
            has_signal=any(mode in query.modes for query in queries),
        )
