from __future__ import annotations

import copy
import unicodedata
from collections.abc import Iterable, Sequence
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
# 같은 대상이 좋아하는 쪽과 싫어하는 쪽에 동시에 남지 않게 한다.
POSITIVE_FIELDS = {
    TasteField.INTERESTS,
    TasteField.HOBBIES,
    TasteField.PREFERENCES,
    TasteField.WANTS,
    TasteField.UNAFFORDABLE,
    TasteField.CONSUMABLES,
}
NEGATIVE_FIELDS = SAFETY_FIELDS

# 대화에서 모으는 취향은 최대 5개이고, 이 안에서 taste_rank_score 순으로 순위를 매긴다.
MAX_ACTIVE_QUERY_SIGNALS = 5
MAX_ACTIVE_WEIGHT_SIGNALS = 3
REPETITION_BONUS = 0.05
HOBBY_BONUS = 0.05
MAX_ACTIVE_OWNED_SIGNALS = 3
MAX_AXES = 3

# 일반적인 응답을 걸러내는 1차 책임은 EXTRACTION_SYSTEM 프롬프트와
# noneAnswer 경로에 있다. 아래 목록은 그 경로가 실패했을 때
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


def _grounded_turn(
    evidence: str,
    utterance: str,
    source_turn: int,
    context_utterances: Sequence[tuple[int, str]],
) -> int | None:
    """근거가 나온 사용자 턴 번호. 이번 발화를 먼저 보고, 없으면 최근 발화를 최신순으로 본다."""
    if _is_grounded(evidence, utterance):
        return source_turn
    for turn, text in reversed(context_utterances):
        if _is_grounded(evidence, text):
            return turn
    return None


def taste_rank_score(signal: ProfileSignal) -> float:
    """취향 순위 점수(0~1). 신뢰도에 여러 번 언급된 것과 반복 활동(hobbies)을 더 얹는다."""
    repetition_bonus = min(signal.mention_count - 1, 3) * REPETITION_BONUS
    hobby_bonus = HOBBY_BONUS if signal.field == TasteField.HOBBIES else 0.0
    return round(min(signal.confidence + repetition_bonus + hobby_bonus, 1.0), 2)


def _signal_sort_key(signal: ProfileSignal) -> tuple[float, ...]:
    specificity = min(len(signal.normalized_value), 20) / 20
    return (
        taste_rank_score(signal),
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


def _polarity(field: TasteField) -> str:
    """좋아하는 쪽, 싫어하는 쪽, 그 밖의 칸. 취소한 값이 같은 쪽으로 되살아나는지 볼 때 쓴다."""
    if field in POSITIVE_FIELDS:
        return "positive"
    if field in NEGATIVE_FIELDS:
        return "negative"
    return field.value


def _live_signals(
    signals: Iterable[ProfileSignal],
    fields: set[TasteField],
    normalized: str,
) -> list[ProfileSignal]:
    return [
        signal
        for signal in signals
        if signal.field in fields
        and signal.normalized_value == normalized
        and signal.status != SignalStatus.SUPERSEDED
    ]


def _group_key(signal: ProfileSignal) -> tuple[str, ...]:
    return signal.taxonomy_path or (signal.field.value, signal.normalized_value)


def rank_by_group_strength(
    signals: Iterable[ProfileSignal],
    *,
    anchors: frozenset[str] = frozenset(),
) -> list[ProfileSignal]:
    """단서가 많이 쌓인 분류(취향 축, 관심사 분야)를 앞에 두고 분류마다 하나씩 돌아가며 뽑는다.

    분류의 강도는 그 분류에 모인 언급 수에, 서로 다른 관심사에 걸쳐 나온 만큼을 더한 값이다.
    "혼자 가는 캠핑"과 "혼자 보는 영화"가 함께 있으면 사회/인원 축이 한 번 나온 축보다 앞선다.
    분류 안에서는 노출된 관심사(anchors)에 붙은 항목을 먼저, 그다음 순위 점수 순으로 둔다.
    """
    groups: dict[tuple[str, ...], list[ProfileSignal]] = {}
    for signal in signals:
        groups.setdefault(_group_key(signal), []).append(signal)

    def member_key(signal: ProfileSignal) -> tuple[float, ...]:
        anchored = bool(signal.target) and normalize_text(signal.target or "") in anchors
        return (float(anchored), *_signal_sort_key(signal))

    ranked_groups: list[tuple[int, tuple[float, ...], list[ProfileSignal]]] = []
    for members in groups.values():
        members.sort(key=member_key, reverse=True)
        mentions = sum(member.mention_count for member in members)
        targets = {normalize_text(member.target) for member in members if member.target}
        strength = mentions + max(len(targets) - 1, 0)
        ranked_groups.append((strength, _signal_sort_key(members[0]), members))
    ranked_groups.sort(key=lambda group: (group[0], group[1]), reverse=True)

    ordered: list[ProfileSignal] = []
    depth = 0
    while len(ordered) < sum(len(members) for _, _, members in ranked_groups):
        for _, _, members in ranked_groups:
            if depth < len(members):
                ordered.append(members[depth])
        depth += 1
    return ordered


def _apply_working_set(signals: list[ProfileSignal]) -> None:
    eligible = [signal for signal in signals if signal.status != SignalStatus.SUPERSEDED]
    for signal in eligible:
        signal.status = SignalStatus.INACTIVE

    def activate_top(
        candidates: list[ProfileSignal],
        limit: int | None,
        *,
        anchors: frozenset[str] = frozenset(),
    ) -> list[ProfileSignal]:
        ordered = rank_by_group_strength(candidates, anchors=anchors)
        selected = ordered if limit is None else ordered[:limit]
        for signal in selected:
            signal.status = SignalStatus.ACTIVE
        return selected

    interests = activate_top(
        [signal for signal in eligible if signal.field in QUERY_FIELDS],
        MAX_ACTIVE_QUERY_SIGNALS,
    )
    # 취향은 노출된 관심사에 붙은 것을 대표로 고른다. 추천이 그 관심사 상품을 찾을 때 바로 쓰인다.
    activate_top(
        [signal for signal in eligible if signal.field in WEIGHT_FIELDS],
        MAX_ACTIVE_WEIGHT_SIGNALS,
        anchors=frozenset(signal.normalized_value for signal in interests),
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
        context_utterances: Sequence[tuple[int, str]] = (),
    ) -> MergeResult:
        merged = copy.deepcopy(profile)
        accepted: list[ProfileSignal] = []
        rejected: list[str] = []

        dropped_now: set[tuple[str, str]] = set()
        for drop in delta.drop:
            normalized = normalize_text(drop.value)
            dropped_now.add((_polarity(drop.field), normalized))
            for signal in merged.signals:
                if signal.field == drop.field and signal.normalized_value == normalized:
                    signal.status = SignalStatus.SUPERSEDED
        # 대체된 값. 이전 발화 문맥만으로는 되살리지 않는다.
        superseded_values = {
            signal.normalized_value
            for signal in merged.signals
            if signal.status == SignalStatus.SUPERSEDED
        }

        # 한 턴에서 같은 대상이 싫은 것과 좋은 것으로 함께 나오면 싫다는 쪽을 믿는다.
        negative_in_turn = {
            normalize_text(item.value) for item in delta.items if item.field in NEGATIVE_FIELDS
        }

        for item in delta.items:
            normalized = normalize_text(item.value)
            item_turn = _grounded_turn(item.evidence, utterance, source_turn, context_utterances)
            if not normalized or normalized in _GENERIC_VALUES or item_turn is None:
                rejected.append(item.value)
                continue
            # 이번 턴에 취소한 값이 같은 쪽으로 다시 들어오거나
            # ("캠핑은 취소"에서 캠핑을 또 뽑은 경우), 취소·대체된 값이
            # 이전 발화 근거로 되살아나는 것을 막는다.
            if (_polarity(item.field), normalized) in dropped_now or (
                item_turn != source_turn and normalized in superseded_values
            ):
                rejected.append(item.value)
                continue
            if item.field in POSITIVE_FIELDS:
                if normalized in negative_in_turn:
                    rejected.append(item.value)
                    continue
                # 싫다고 했던 대상은 이번 발화에서 직접 좋다고 말했을 때만 마음이 바뀐 것으로 본다.
                negatives = _live_signals(merged.signals, NEGATIVE_FIELDS, normalized)
                if negatives:
                    if item.evidence_type != EvidenceType.EXPLICIT or item_turn != source_turn:
                        rejected.append(item.value)
                        continue
                    for signal in negatives:
                        signal.status = SignalStatus.SUPERSEDED
            elif item.field in NEGATIVE_FIELDS:
                for signal in _live_signals(merged.signals, POSITIVE_FIELDS, normalized):
                    signal.status = SignalStatus.SUPERSEDED
            # 이전 턴 근거는 새 항목을 만들 때만 쓴다. 문맥에 다시 보인다는 이유로
            # 이미 있는 항목의 언급 횟수를 올리면 반복이 부풀려진다.
            if item_turn != source_turn and any(
                signal.field == item.field
                and signal.normalized_value == normalized
                and signal.status != SignalStatus.SUPERSEDED
                for signal in merged.signals
            ):
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
                existing.source_turn = item_turn
                existing.aspect = item.aspect or existing.aspect
                existing.target = item.target or existing.target
                existing.taxonomy_path = item.taxonomy_path or existing.taxonomy_path
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
                aspect=item.aspect,
                target=item.target,
                taxonomy_path=item.taxonomy_path,
                first_seen_at=now,
                updated_at=now,
                source_turn=item_turn,
            )
            merged.signals.append(signal)
            accepted.append(signal)

        for axis in delta.axes:
            cleaned = " ".join(axis.strip().split())
            normalized = normalize_text(cleaned)
            if (
                not cleaned
                or _grounded_turn(cleaned, utterance, source_turn, context_utterances) is None
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


def ranked_friend_signals(
    profile: ProfileState,
    fields: Iterable[TasteField],
    *,
    include_inactive: bool = False,
) -> list[ProfileSignal]:
    """친구에게 보여도 되는 신호를 working set 과 같은 기준의 우선순위 순으로 돌려준다.

    include_inactive면 활성 한도 밖으로 밀린 저장 항목까지 포함한다. 요약문이 이 범위를 쓴다.
    """
    allowed = set(fields)
    pool = profile.stored_signals() if include_inactive else profile.active_signals()
    anchors = frozenset(
        signal.normalized_value
        for signal in profile.active_signals()
        if signal.field in QUERY_FIELDS
    )
    ordered = rank_by_group_strength(
        (
            signal
            for signal in pool
            if signal.field in allowed and signal.visibility == Visibility.FRIENDS
        ),
        anchors=anchors,
    )
    ranked: list[ProfileSignal] = []
    seen: set[str] = set()
    for signal in ordered:
        if signal.normalized_value in seen:
            continue
        seen.add(signal.normalized_value)
        ranked.append(signal)
    return ranked


def friend_signal_groups(
    profile: ProfileState,
    fields: Iterable[TasteField],
    *,
    limit: int,
) -> list[tuple[tuple[str, ...], list[ProfileSignal]]]:
    """요약문용. 저장된 친구 공개 항목을 강도 순으로 limit개까지 고르고 분류별로 묶는다."""
    groups: dict[tuple[str, ...], list[ProfileSignal]] = {}
    for signal in ranked_friend_signals(profile, fields, include_inactive=True)[:limit]:
        groups.setdefault(_group_key(signal), []).append(signal)
    return list(groups.items())
