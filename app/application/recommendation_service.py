from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.application.recommendation_engine import RecommendationEngine
from app.domain.profile.models import DeferralReason, TasteField, Visibility
from app.domain.recommendation import (
    RankedRecommendation,
    RecommendationGuide,
    RecommendationSignal,
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
                    rank_score=(
                        float(item["rankScore"]) if item.get("rankScore") is not None else None
                    ),
                )
            )
    return tuple(signals)


def _values(profile: Mapping[str, Any], fields: Sequence[TasteField]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        items = profile.get(field.value, [])
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        values.extend(
            str(item["value"]) for item in items if isinstance(item, Mapping) and item.get("value")
        )
    return tuple(values)


def _recommendation_guide(payload: Mapping[str, Any]) -> RecommendationGuide:
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        return RecommendationGuide()
    return RecommendationGuide(
        exclusions=_values(profile, (TasteField.DISLIKES, TasteField.CONSTRAINTS)),
        preferences=_values(profile, (TasteField.PREFERENCES,)),
    )


def _items(recommendations: Sequence[RankedRecommendation]) -> list[dict[str, object]]:
    return [
        {
            "platform": recommendation.product.key.platform,
            "externalId": recommendation.product.key.external_id,
            "score": round(recommendation.score, 2),
            "reason": recommendation.reason,
        }
        for recommendation in recommendations
    ]


class V1RecommendationService:
    def __init__(self, engine: RecommendationEngine) -> None:
        self._engine = engine

    async def recommend_lists(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        signals = _recommendation_signals(payload)
        batch = await self._engine.recommend(signals, guide=_recommendation_guide(payload))
        return {
            "generatedAt": datetime.now(UTC).isoformat(),
            "self": {"items": _items(batch.self)},
            "gift": {"items": _items(batch.gift)},
        }
