from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.application.recommendation_engine import RecommendationEngine
from app.domain.profile.models import DeferralReason, PreferenceAspect, TasteField, Visibility
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
# content 유사도 가산에 쓰는 취향 측면. 상황은 보조 usage 쿼리로 쓰고,
# 동기는 상품 문서와 맞지 않아 추천에 쓰지 않는다. 측면이 없는 이전 세션 값은 가산에 쓴다.
_BOOST_ASPECTS = {
    None,
    PreferenceAspect.ATTRIBUTE,
    PreferenceAspect.SENSORY,
    PreferenceAspect.CRITERION,
}


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("profile updatedAt must be a datetime")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _aspect(item: Mapping[str, Any]) -> PreferenceAspect | None:
    value = item.get("aspect")
    try:
        return PreferenceAspect(str(value)) if value else None
    except ValueError:
        return None


def _taxonomy_path(item: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = item.get("taxonomyPath")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return tuple(str(part) for part in value)
    return None


def _preference_signals(profile: Mapping[str, Any]) -> list[RecommendationSignal]:
    """취향은 관심사와 조합한 쿼리와 상황 쿼리의 재료가 된다."""
    items = profile.get(TasteField.PREFERENCES.value, [])
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return []
    return [
        RecommendationSignal(
            field=TasteField.PREFERENCES,
            value=str(item["value"]),
            confidence=float(item["confidence"]),
            visibility=Visibility(str(item.get("visibility", Visibility.FRIENDS))),
            updated_at=_parse_datetime(item["updatedAt"]),
            rank_score=(float(item["rankScore"]) if item.get("rankScore") is not None else None),
            aspect=_aspect(item),
            target=str(item["target"]) if item.get("target") else None,
            taxonomy_path=_taxonomy_path(item),
        )
        for item in items
        # 신뢰도나 갱신 시각이 없는 항목은 쿼리 재료로 쓰지 않고 가산에만 쓴다.
        if isinstance(item, Mapping)
        and item.get("value")
        and item.get("confidence") is not None
        and item.get("updatedAt")
    ]


def recommendation_inputs(
    payload: Mapping[str, Any],
) -> tuple[tuple[RecommendationSignal, ...], RecommendationGuide]:
    """추천 엔진에 넘기는 신호와 가이드. 플레이그라운드도 같은 함수로 쿼리를 미리 본다."""
    return _recommendation_signals(payload), _recommendation_guide(payload)


def _recommendation_signals(payload: Mapping[str, Any]) -> tuple[RecommendationSignal, ...]:
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("profile must be an object")

    signals: list[RecommendationSignal] = _preference_signals(profile)
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
                    taxonomy_path=_taxonomy_path(item),
                )
            )
    return tuple(signals)


def _values(
    profile: Mapping[str, Any],
    fields: Sequence[TasteField],
    *,
    aspects: set[PreferenceAspect | None] | None = None,
) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        items = profile.get(field.value, [])
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        values.extend(
            str(item["value"])
            for item in items
            if isinstance(item, Mapping)
            and item.get("value")
            and (aspects is None or _aspect(item) in aspects)
        )
    return tuple(values)


def _recommendation_guide(payload: Mapping[str, Any]) -> RecommendationGuide:
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        return RecommendationGuide()
    return RecommendationGuide(
        exclusions=_values(profile, (TasteField.DISLIKES, TasteField.CONSTRAINTS)),
        preferences=_values(profile, (TasteField.PREFERENCES,), aspects=_BOOST_ASPECTS),
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
        signals, guide = recommendation_inputs(payload)
        batch = await self._engine.recommend(signals, guide=guide)
        return {
            "generatedAt": datetime.now(UTC).isoformat(),
            "self": {"items": _items(batch.self)},
            "gift": {"items": _items(batch.gift)},
        }
