from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.application.recommendation_engine import RecommendationEngine
from app.application.recommendation_reason_service import RecommendationReasonService
from app.application.search_intent_service import SearchIntentService, intent_targets
from app.domain.profile.models import DeferralReason, PreferenceAspect, TasteField, Visibility
from app.domain.profile.onboarding import (
    INTEREST_QUERY_TERMS,
    INTEREST_TAXONOMY_KEYS,
    Allergy,
    GiftExclusion,
    InterestCategory,
)
from app.domain.profile.taxonomy import INTEREST_CATEGORIES_BY_KEY
from app.domain.recommendation import (
    ProductKey,
    RankedRecommendation,
    RecommendationGuide,
    RecommendationSignal,
    build_catalog_exclusion,
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


# 반려동물 분야. 어떤 항목이든 이 분야로 분류됐으면 반려동물과 관련된 사용자로 본다.
_PET_INTEREST_PATH = INTEREST_CATEGORIES_BY_KEY["pets"].path
# 반려동물 신호로 보지 않는 필드. 싫거나 못 쓰는 것은 반려동물을 원한다는 뜻이 아니다.
_NOT_PET_EVIDENCE = frozenset({TasteField.DISLIKES, TasteField.CONSTRAINTS})

# 온보딩 관심 카테고리의 쿼리 가중치. 쿼리 재료로 늘 쓰이도록 강한 신호 기준(0.55)은 넘기되,
# 대화에서 구체적으로 나온 관심사(대개 0.7 이상)보다 앞서지 않게 둔다.
ONBOARDING_INTEREST_WEIGHT = 0.6


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
    *,
    now: datetime | None = None,
) -> tuple[tuple[RecommendationSignal, ...], RecommendationGuide]:
    """추천 엔진에 넘기는 신호와 가이드. 플레이그라운드도 같은 함수로 쿼리를 미리 본다.

    payload 의 onboarding 은 선택이다. 관심 카테고리는 신호가 되고, 알레르기와 선물 제외
    조건은 가이드의 catalog_exclusion 이 된다.
    """
    onboarding = payload.get("onboarding") or {}
    if not isinstance(onboarding, Mapping):
        raise ValueError("onboarding must be an object")
    signals = (
        *_recommendation_signals(payload),
        *_onboarding_signals(onboarding, now or datetime.now(UTC)),
    )
    return signals, _recommendation_guide(payload, onboarding)


def _codes(onboarding: Mapping[str, Any], key: str, enum: type[Any]) -> tuple[Any, ...]:
    """코드 배열을 enum 으로 바꾼다. 같은 코드가 두 번 와도 신호는 하나만 만든다."""
    values = onboarding.get(key, [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"onboarding {key} must be an array")
    return tuple(dict.fromkeys(enum(str(value)) for value in values))


def _onboarding_signals(
    onboarding: Mapping[str, Any],
    now: datetime,
) -> list[RecommendationSignal]:
    """사용자가 고른 관심 카테고리. 친구에게 보이는 화면 칩이라 공개 범위는 친구로 둔다."""
    return [
        RecommendationSignal(
            field=TasteField.INTERESTS,
            value=INTEREST_QUERY_TERMS[interest],
            confidence=ONBOARDING_INTEREST_WEIGHT,
            visibility=Visibility.FRIENDS,
            updated_at=now,
            taxonomy_path=INTEREST_CATEGORIES_BY_KEY[INTEREST_TAXONOMY_KEYS[interest]].path,
        )
        for interest in _codes(onboarding, "interestCategoryCodes", InterestCategory)
    ]


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


def _knows_pets(payload: Mapping[str, Any], onboarding: Mapping[str, Any]) -> bool:
    """반려동물과 관련된 사용자인지. 온보딩에서 반려동물을 골랐거나, 프로필 항목 중 하나라도
    반려동물 분야로 분류됐으면 그렇다고 본다. 단어 목록이 아니라 판단 모델의 분야 분류를 쓴다.
    """
    if InterestCategory.PET in _codes(onboarding, "interestCategoryCodes", InterestCategory):
        return True
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        return False
    for field in TasteField:
        if field in _NOT_PET_EVIDENCE:
            continue
        items = profile.get(field.value, [])
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            continue
        if any(
            isinstance(item, Mapping) and _taxonomy_path(item) == _PET_INTEREST_PATH
            for item in items
        ):
            return True
    return False


def _recommendation_guide(
    payload: Mapping[str, Any],
    onboarding: Mapping[str, Any],
) -> RecommendationGuide:
    catalog_exclusion = build_catalog_exclusion(
        allergies=_codes(onboarding, "allergyCodes", Allergy),
        gift_exclusions=_codes(onboarding, "giftExclusionCodes", GiftExclusion),
        exclude_pets=not _knows_pets(payload, onboarding),
    )
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        return RecommendationGuide(catalog_exclusion=catalog_exclusion)
    return RecommendationGuide(
        exclusions=_values(profile, (TasteField.DISLIKES, TasteField.CONSTRAINTS)),
        preferences=_values(profile, (TasteField.PREFERENCES,), aspects=_BOOST_ASPECTS),
        catalog_exclusion=catalog_exclusion,
    )


def _recipient_name(payload: Mapping[str, Any]) -> str | None:
    """선물 추천 이유에 넣을 이름. 대화방을 만들 때 받은 username 을 정리한 값이다."""
    onboarding = payload.get("onboarding")
    if not isinstance(onboarding, Mapping):
        return None
    name = onboarding.get("displayName")
    return str(name) if name else None


def _items(
    recommendations: Sequence[RankedRecommendation],
    reasons: Mapping[ProductKey, str],
) -> list[dict[str, object]]:
    """reasons 는 다시 쓴 추천 이유다. 없는 상품은 순위를 매길 때 붙은 템플릿 이유를 쓴다."""
    return [
        {
            "platform": recommendation.product.key.platform,
            "externalId": recommendation.product.key.external_id,
            "score": round(recommendation.score, 2),
            "reason": reasons.get(recommendation.product.key, recommendation.reason),
        }
        for recommendation in recommendations
    ]


class V1RecommendationService:
    def __init__(
        self,
        engine: RecommendationEngine,
        *,
        intents: SearchIntentService | None = None,
        reasons: RecommendationReasonService | None = None,
    ) -> None:
        """reasons 가 있으면 추천 이유를 상품에 맞는 문장으로 다시 쓴다."""
        self._engine = engine
        self._intents = intents
        self._reasons = reasons

    async def recommend_lists(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        signals, guide = recommendation_inputs(payload)
        intents = {}
        if self._intents is not None:
            # 추천은 사용자가 기다리는 경로라 캐시만 읽는다. 없는 의도는 다음 추천을 위해
            # 백그라운드로 만들고, 이번에는 템플릿 쿼리로 찾는다.
            targets = intent_targets(signals)
            intents = dict(await self._intents.lookup(targets))
            self._intents.schedule(target for target in targets if target.key not in intents)
        recipient_name = _recipient_name(payload)
        batch = await self._engine.recommend(
            signals,
            guide=guide,
            recipient_name=recipient_name,
            intents=intents,
        )
        self_reasons: Mapping[ProductKey, str] = {}
        gift_reasons: Mapping[ProductKey, str] = {}
        if self._reasons is not None:
            self_reasons, gift_reasons = await asyncio.gather(
                self._reasons.rewrite(batch.self),
                self._reasons.rewrite(batch.gift, recipient_name=recipient_name),
            )
        return {
            "generatedAt": datetime.now(UTC).isoformat(),
            "self": {"items": _items(batch.self, self_reasons)},
            "gift": {"items": _items(batch.gift, gift_reasons)},
        }
