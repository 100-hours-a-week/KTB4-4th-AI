from __future__ import annotations

import copy
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from app.domain.profile.models import (
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
    IntentType,
    LinkRole,
    ProfileSignal,
    ProfileState,
    SignalStatus,
    TasteField,
    Visibility,
)

QUERY_FIELDS = {
    TasteField.INTERESTS,
    TasteField.HOBBIES,
    TasteField.WANTS,
    TasteField.UNAFFORDABLE,
    TasteField.CONSUMABLES,
}
WEIGHT_FIELDS = {TasteField.PREFERENCES, TasteField.LIFESTYLE}
SAFETY_FIELDS = {TasteField.DISLIKES, TasteField.CONSTRAINTS}

MAX_ACTIVE_QUERY_SIGNALS = 6
MAX_ACTIVE_WEIGHT_SIGNALS = 3
MAX_ACTIVE_OWNED_SIGNALS = 3
MAX_AXES = 3

# 일반적인 응답을 걸러내는 1차 책임은 EXTRACTION_SYSTEM 프롬프트와
# goalAssessment.confirmed_none 경로에 있다. 아래 목록은 그 경로가 실패했을 때
# 프로필이 오염되지 않게 막는 최소한의 안전망이므로, 새 사례가 나와도
# 여기에 단어를 추가하지 말고 프롬프트를 고친다.
_GENERIC_VALUES = frozenset(
    {
        "그냥",
        "아무거나",
        "잘 모르겠어요",
        "잘 모르겠어",
        "모르겠어요",
        "모르겠어",
        "보통",
        "다 좋아요",
        "다 좋아",
        "상관없어요",
        "상관없어",
        "글쎄요",
        "글쎄",
    }
)

_CONFIDENCE_CEILING = {
    EvidenceType.EXPLICIT: 1.0,
    EvidenceType.CONFIRMED: 0.9,
    EvidenceType.INFERRED: 0.65,
}

_EVIDENCE_PRIORITY = {
    EvidenceType.EXPLICIT: 3,
    EvidenceType.CONFIRMED: 2,
    EvidenceType.INFERRED: 1,
}


@dataclass(slots=True, frozen=True)
class MergeResult:
    profile: ProfileState
    accepted: tuple[ProfileSignal, ...]
    rejected_values: tuple[str, ...]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.strip().split()).lower()


def link_role_for(field: TasteField) -> LinkRole:
    if field in QUERY_FIELDS:
        return LinkRole.QUERY
    if field in WEIGHT_FIELDS:
        return LinkRole.WEIGHT
    return LinkRole.FILTER


def visibility_for(field: TasteField) -> Visibility:
    if field in {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.PREFERENCES,
    }:
        return Visibility.FRIENDS
    return Visibility.PRIVATE


def default_intent_for(field: TasteField) -> IntentType | None:
    if field in {TasteField.WANTS, TasteField.UNAFFORDABLE}:
        return IntentType.WANT
    if field == TasteField.CONSUMABLES:
        return IntentType.NEED
    if field in {TasteField.INTERESTS, TasteField.HOBBIES}:
        return IntentType.BOTH
    return None


def _is_grounded(evidence: str, utterance: str) -> bool:
    normalized_evidence = normalize_text(evidence)
    return bool(normalized_evidence) and normalized_evidence in normalize_text(utterance)


def _signal_sort_key(signal: ProfileSignal) -> tuple[float, ...]:
    specificity = min(len(signal.normalized_value), 20) / 20
    return (
        signal.confidence,
        float(_EVIDENCE_PRIORITY[signal.evidence_type]),
        float(signal.mention_count),
        specificity,
        signal.updated_at.timestamp(),
    )


def _supersede_previous_state(signals: list[ProfileSignal], item: ExtractedItem) -> None:
    transitions: dict[TasteField, set[TasteField]] = {
        TasteField.UNAFFORDABLE: {TasteField.WANTS},
        TasteField.WANTS: {TasteField.UNAFFORDABLE},
        TasteField.OWNED: {TasteField.WANTS, TasteField.UNAFFORDABLE},
    }
    previous_fields = transitions.get(item.field, set())
    if not previous_fields:
        return
    normalized = normalize_text(item.value)
    for signal in signals:
        if signal.field in previous_fields and signal.normalized_value == normalized:
            signal.status = SignalStatus.SUPERSEDED


def _apply_working_set(signals: list[ProfileSignal]) -> None:
    eligible = [signal for signal in signals if signal.status != SignalStatus.SUPERSEDED]
    for signal in eligible:
        signal.status = SignalStatus.INACTIVE

    def activate_top(candidates: list[ProfileSignal], limit: int | None) -> None:
        ordered = sorted(candidates, key=_signal_sort_key, reverse=True)
        selected = ordered if limit is None else ordered[:limit]
        for signal in selected:
            signal.status = SignalStatus.ACTIVE

    activate_top(
        [signal for signal in eligible if signal.field in QUERY_FIELDS],
        MAX_ACTIVE_QUERY_SIGNALS,
    )
    activate_top(
        [signal for signal in eligible if signal.field in WEIGHT_FIELDS],
        MAX_ACTIVE_WEIGHT_SIGNALS,
    )
    activate_top(
        [signal for signal in eligible if signal.field == TasteField.OWNED],
        MAX_ACTIVE_OWNED_SIGNALS,
    )
    activate_top([signal for signal in eligible if signal.field in SAFETY_FIELDS], None)


class ProfileMerger:
    def merge(
        self,
        profile: ProfileState,
        delta: ExtractionDelta,
        *,
        utterance: str,
        now: datetime,
        source_turn: int,
    ) -> MergeResult:
        merged = copy.deepcopy(profile)
        accepted: list[ProfileSignal] = []
        rejected: list[str] = []

        for drop in delta.drop:
            normalized = normalize_text(drop.value)
            for signal in merged.signals:
                if signal.field == drop.field and signal.normalized_value == normalized:
                    signal.status = SignalStatus.SUPERSEDED

        for item in delta.items:
            normalized = normalize_text(item.value)
            if (
                not normalized
                or normalized in _GENERIC_VALUES
                or not _is_grounded(item.evidence, utterance)
            ):
                rejected.append(item.value)
                continue

            _supersede_previous_state(merged.signals, item)
            calibrated_confidence = min(
                max(item.confidence, 0.0),
                _CONFIDENCE_CEILING[item.evidence_type],
            )
            existing = next(
                (
                    signal
                    for signal in merged.signals
                    if signal.field == item.field
                    and signal.normalized_value == normalized
                    and signal.status != SignalStatus.SUPERSEDED
                ),
                None,
            )
            if existing is not None:
                existing.confidence = max(existing.confidence, calibrated_confidence)
                existing.updated_at = now
                existing.mention_count += 1
                existing.evidence = item.evidence
                if (
                    _EVIDENCE_PRIORITY[item.evidence_type]
                    >= _EVIDENCE_PRIORITY[existing.evidence_type]
                ):
                    existing.evidence_type = item.evidence_type
                existing.intent_type = item.intent_type or existing.intent_type
                existing.deferral_reason = item.deferral_reason or existing.deferral_reason
                existing.value = item.value.strip()
                existing.source_turn = source_turn
                existing.status = SignalStatus.ACTIVE
                accepted.append(existing)
                continue

            signal = ProfileSignal(
                field=item.field,
                value=item.value.strip(),
                normalized_value=normalized,
                confidence=calibrated_confidence,
                link_role=link_role_for(item.field),
                visibility=visibility_for(item.field),
                intent_type=item.intent_type or default_intent_for(item.field),
                deferral_reason=item.deferral_reason,
                evidence=item.evidence,
                evidence_type=item.evidence_type,
                first_seen_at=now,
                updated_at=now,
                source_turn=source_turn,
            )
            merged.signals.append(signal)
            accepted.append(signal)

        for axis in delta.axes:
            cleaned = " ".join(axis.strip().split())
            normalized = normalize_text(cleaned)
            if (
                not cleaned
                or not _is_grounded(cleaned, utterance)
                or normalized in {normalize_text(value) for value in merged.axes}
            ):
                continue
            if len(merged.axes) < MAX_AXES:
                merged.axes.append(cleaned)

        _apply_working_set(merged.signals)
        return MergeResult(
            profile=merged,
            accepted=tuple(accepted),
            rejected_values=tuple(rejected),
        )


def friend_summary_values(profile: ProfileState) -> dict[str, list[str]]:
    allowed = (
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.PREFERENCES,
    )
    values: dict[str, list[str]] = {field.value: [] for field in allowed}
    for signal in profile.active_signals():
        if signal.field in allowed and signal.visibility == Visibility.FRIENDS:
            values[signal.field.value].append(signal.value)
    return values
