"""대화방 사이에 이어지는 장기 기억.

AI 서버는 기억을 저장하지 않는다. 대화를 닫을 때 이번 대화 결과를 기존 기억에 합쳐 돌려주고,
Backend가 그대로 저장했다가 다음 대화방을 만들 때 다시 보낸다.
진행률과 종료 판정은 이번 대화에서 모은 것만 센다. 기억은 첫인사와 질문 조절,
분석·추천 프로필에만 쓴다.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.domain.profile.merger import (
    EVIDENCE_PRIORITY,
    MAX_AXES,
    PREVIOUS_STATES,
    SAFETY_FIELDS,
    apply_working_set,
    normalize_text,
    opposite_fields,
    taste_rank_score,
)
from app.domain.profile.models import (
    DropRef,
    InsightKind,
    ProfileSignal,
    ProfileState,
    SignalStatus,
    TasteField,
)

MEMORY_SCHEMA_VERSION = "1.0"
# 마지막으로 확인한 뒤 이만큼 지나면 확신도가 절반이 된다.
HALF_LIFE_DAYS = 90.0
# 확신도가 이보다 낮아지면 잊는다.
FORGET_CONFIDENCE = 0.3
# 이 이상이면 이미 아는 것으로 보고 대화에서 다시 묻지 않는다.
# 잊을 만큼은 아니고 이보다 낮은 기억은 대화 중 한 번 다시 확인할 후보다.
KNOWN_CONFIDENCE = 0.6
# 싫어하는 것과 피해야 하는 것은 시간이 지나도 낮추지 않는다. 잊어서 다시 추천되는 쪽이 더 위험하다.
NON_DECAYING_FIELDS = SAFETY_FIELDS
MAX_MEMORY_SIGNALS = 40
# 최근 대화에서 니쥬가 먼저 꺼낸 기억. 다음 첫인사에서 같은 이야기를 반복하지 않는 데 쓴다.
RECALLED_LIMIT = 3
# 니쥬가 먼저 꺼내는 기억. 갖고 싶지만 못 사는 것 같은 사정이나 싫어하는 것은 먼저 꺼내지 않는다.
RECALLABLE_FIELDS = frozenset({TasteField.INTERESTS, TasteField.HOBBIES})
# 마지막 대화 뒤 이만큼 지났으면 첫인사에서 오랜만이라고 한다.
LONG_ABSENCE_DAYS = 30

MemoryKey = tuple[TasteField, str]


@dataclass(slots=True)
class LongTermMemory:
    user_id: int
    signals: list[ProfileSignal] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)
    session_count: int = 0
    updated_at: datetime | None = None
    recalled: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "signals": [signal.to_dict() for signal in self.signals],
            "axes": list(self.axes),
            "session_count": self.session_count,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "recalled": list(self.recalled),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LongTermMemory:
        updated_at = payload.get("updated_at")
        return cls(
            user_id=int(payload["user_id"]),
            signals=[ProfileSignal.from_dict(item) for item in payload.get("signals", [])],
            axes=list(payload.get("axes", [])),
            session_count=int(payload.get("session_count", 0)),
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
            recalled=list(payload.get("recalled", [])),
        )


def memory_key(signal: ProfileSignal) -> MemoryKey:
    return signal.field, signal.normalized_value


def effective_confidence(signal: ProfileSignal, now: datetime) -> float:
    """마지막으로 확인한 때(updated_at)부터 지난 시간만큼 낮춘 확신도."""
    if signal.field in NON_DECAYING_FIELDS:
        return signal.confidence
    age_days = max((now - signal.updated_at).total_seconds(), 0.0) / 86400.0
    return signal.confidence * 0.5 ** (age_days / HALF_LIFE_DAYS)


def _remember(signal: ProfileSignal, now: datetime) -> ProfileSignal | None:
    """지금 기준 확신도로 낮춘 사본. 잊을 만큼 낮으면 None 이다."""
    confidence = effective_confidence(signal, now)
    if confidence < FORGET_CONFIDENCE:
        return None
    return replace(signal, confidence=round(confidence, 3), status=SignalStatus.ACTIVE)


def remembered_signals(memory: LongTermMemory | None, now: datetime) -> list[ProfileSignal]:
    """잊지 않은 기억의 사본."""
    if memory is None:
        return []
    return [
        remembered
        for remembered in (_remember(signal, now) for signal in memory.signals)
        if remembered is not None
    ]


def dropped_memory_keys(memory: LongTermMemory | None, drops: Iterable[DropRef]) -> list[MemoryKey]:
    """이번 턴에 취소한 값 중 기억에 있는 것. 세션 프로필에 없던 기억도 닫을 때 지운다."""
    if memory is None:
        return []
    known = {memory_key(signal) for signal in memory.signals}
    keys = [(drop.field, normalize_text(drop.value)) for drop in drops]
    return [key for key in dict.fromkeys(keys) if key in known]


def _refresh(remembered: ProfileSignal, current: ProfileSignal) -> None:
    """이번 대화에서 다시 나온 기억. 확인한 시각을 새로 하고 이번 대화의 표현을 따른다."""
    remembered.confidence = max(remembered.confidence, current.confidence)
    remembered.mention_count += current.mention_count
    remembered.updated_at = current.updated_at
    remembered.value = current.value
    remembered.evidence = current.evidence
    if EVIDENCE_PRIORITY[current.evidence_type] >= EVIDENCE_PRIORITY[remembered.evidence_type]:
        remembered.evidence_type = current.evidence_type
    remembered.intent_type = current.intent_type or remembered.intent_type
    remembered.deferral_reason = current.deferral_reason or remembered.deferral_reason
    remembered.aspect = current.aspect or remembered.aspect
    remembered.target = current.target or remembered.target
    remembered.taxonomy_path = current.taxonomy_path or remembered.taxonomy_path
    remembered.source_turn = current.source_turn
    if current.insight_kind != InsightKind.PREFERENCE:
        remembered.insight_kind = current.insight_kind
    remembered.context = current.context or remembered.context
    remembered.reason = current.reason or remembered.reason
    remembered.origin = current.origin or remembered.origin


def _combine(
    memory: LongTermMemory | None,
    session: ProfileState,
    *,
    now: datetime,
    dropped: Iterable[MemoryKey],
) -> tuple[ProfileState, dict[int, ProfileSignal]]:
    """기억 위에 이번 대화 결과를 얹는다. 둘이 어긋나면 더 최근인 이번 대화를 따른다.

    두 번째 값은 이번 대화에서 다시 나오지 않은 기억의 사본(id)에서 원본으로 가는 표다.
    저장할 때 원본의 확신도로 되돌려서 시간 감쇠가 겹쳐 걸리지 않게 한다.
    """
    removed = set(dropped)
    # 이번 대화에서 취소되거나 다른 상태로 바뀐 값은 기억에서도 지운다.
    removed.update(
        memory_key(signal) for signal in session.signals if signal.status == SignalStatus.SUPERSEDED
    )
    signals: list[ProfileSignal] = []
    untouched: dict[int, ProfileSignal] = {}
    for original in memory.signals if memory else ():
        remembered = None if memory_key(original) in removed else _remember(original, now)
        if remembered is not None:
            signals.append(remembered)
            untouched[id(remembered)] = original

    for current in session.stored_signals():
        stale = opposite_fields(current.field) | PREVIOUS_STATES.get(current.field, frozenset())
        for signal in [s for s in signals if s.field in stale]:
            if signal.normalized_value == current.normalized_value:
                signals.remove(signal)
                untouched.pop(id(signal), None)
        existing = next((s for s in signals if memory_key(s) == memory_key(current)), None)
        if existing is None:
            signals.append(replace(current))
            continue
        untouched.pop(id(existing), None)
        _refresh(existing, current)

    apply_working_set(signals)
    axes: list[str] = []
    for axis in (*session.axes, *(memory.axes if memory else ())):
        if normalize_text(axis) not in {normalize_text(value) for value in axes}:
            axes.append(axis)
    return ProfileState(signals=signals, axes=axes[:MAX_AXES]), untouched


def combined_profile(
    memory: LongTermMemory | None,
    session: ProfileState,
    *,
    now: datetime,
    dropped: Iterable[MemoryKey] = (),
) -> ProfileState:
    """분석과 추천에 쓰는 프로필. 기억이 없으면 이번 대화 프로필 그대로다."""
    if memory is None or not memory.signals:
        return session
    return _combine(memory, session, now=now, dropped=dropped)[0]


def _trimmed(signals: list[ProfileSignal], now: datetime) -> list[ProfileSignal]:
    """싫어하는 것·피해야 하는 것은 모두 남기고, 나머지는 지금 기준 순위로 상한까지 남긴다."""
    safety = [signal for signal in signals if signal.field in SAFETY_FIELDS]
    others = sorted(
        (signal for signal in signals if signal.field not in SAFETY_FIELDS),
        key=lambda signal: taste_rank_score(
            replace(signal, confidence=effective_confidence(signal, now))
        ),
        reverse=True,
    )
    return [*safety, *others[: max(MAX_MEMORY_SIGNALS - len(safety), 0)]]


def merge_memory(
    memory: LongTermMemory | None,
    session: ProfileState,
    *,
    user_id: int,
    now: datetime,
    dropped: Iterable[MemoryKey] = (),
    recalled: Iterable[str] = (),
) -> LongTermMemory:
    """대화를 닫을 때 Backend에 돌려줄 새 기억."""
    profile, untouched = _combine(memory, session, now=now, dropped=dropped)
    stored: list[ProfileSignal] = []
    for signal in profile.stored_signals():
        kept = replace(signal, status=SignalStatus.ACTIVE, source_turn=0)
        original = untouched.get(id(signal))
        if original is not None:
            kept.confidence = original.confidence
        stored.append(kept)
    return LongTermMemory(
        user_id=user_id,
        signals=_trimmed(stored, now),
        axes=list(profile.axes),
        session_count=(memory.session_count if memory else 0) + 1,
        updated_at=now,
        recalled=list(dict.fromkeys([*recalled, *(memory.recalled if memory else ())]))[
            :RECALLED_LIMIT
        ],
    )


def known_memory(
    memory: LongTermMemory | None,
    now: datetime,
    *,
    limit: int,
) -> list[ProfileSignal]:
    """이미 아는 것으로 보는 기억. 대화에서 다시 묻지 않는 데 쓴다."""
    known = [
        signal
        for signal in remembered_signals(memory, now)
        if signal.confidence >= KNOWN_CONFIDENCE
    ]
    known.sort(key=taste_rank_score, reverse=True)
    return known[:limit]


def _recallable_signal(
    memory: LongTermMemory | None,
    now: datetime,
    *,
    known: bool,
    exclude: Iterable[str],
) -> ProfileSignal | None:
    skipped = {normalize_text(value) for value in exclude}
    candidates = [
        signal
        for signal in remembered_signals(memory, now)
        if signal.field in RECALLABLE_FIELDS
        and (signal.confidence >= KNOWN_CONFIDENCE) == known
        and signal.normalized_value not in skipped
    ]
    if not candidates:
        return None
    return max(candidates, key=taste_rank_score)


def _recallable(
    memory: LongTermMemory | None,
    now: datetime,
    *,
    known: bool,
    exclude: Iterable[str],
) -> str | None:
    signal = _recallable_signal(memory, now, known=known, exclude=exclude)
    return signal.value if signal else None


def opening_topic(memory: LongTermMemory | None, now: datetime) -> str | None:
    """첫인사에서 꺼낼 지난 이야기. 확실한 관심사 중 최근 대화에서 꺼내지 않은 것이다."""
    return _recallable(memory, now, known=True, exclude=memory.recalled if memory else ())


class OpeningAngle(StrEnum):
    """다시 온 사용자에게 지난 이야기를 꺼내는 방식."""

    STATUS = "status"  # 그 활동의 근황
    SCENE = "scene"  # 지난번에 말한 장면에서 이어진 일
    REASON = "reason"  # 같은 이유로 좋았던 다른 것
    ORIGIN = "origin"  # 처음 시작한 때와 지금
    FRESH = "fresh"  # 지난 이야기는 짚기만 하고 새 소재로 묻기


@dataclass(slots=True, frozen=True)
class OpeningHook:
    """첫인사에 쓸 지난 이야기와 꺼내는 방식. quote는 그 방식의 근거가 되는 사용자 발화다."""

    topic: str
    angle: OpeningAngle
    quote: str | None = None
    days_since: int | None = None


# 마지막 대화가 이 안이면 '요즘도'처럼 오래된 일로 묻지 않는다.
RECENT_RETURN_DAYS = 2
_QUOTE_ANGLES = (
    (OpeningAngle.SCENE, "context"),
    (OpeningAngle.REASON, "reason"),
    (OpeningAngle.ORIGIN, "origin"),
)


def days_since_last(memory: LongTermMemory | None, now: datetime) -> int | None:
    if memory is None or memory.updated_at is None:
        return None
    return max((now - memory.updated_at).days, 0)


def opening_angles(signal: ProfileSignal, days_since: int | None) -> list[OpeningAngle]:
    """기억에 실제로 있는 내용으로 꺼낼 수 있는 방식만 고른다.

    방금 대화했으면 근황을 묻지 않고, 오래됐으면 세부를 기억하는 척하지 않는다.
    """
    angles = [OpeningAngle.FRESH]
    if days_since is None or days_since > RECENT_RETURN_DAYS:
        angles.append(OpeningAngle.STATUS)
    if days_since is None or days_since < LONG_ABSENCE_DAYS:
        angles.extend(angle for angle, key in _QUOTE_ANGLES if getattr(signal, key))
    return angles


def opening_hook(
    memory: LongTermMemory | None,
    now: datetime,
    *,
    rng: random.Random | None = None,
) -> OpeningHook | None:
    """첫인사에서 꺼낼 지난 이야기와 방식. 꺼낼 기억이 없으면 None 이다."""
    signal = _recallable_signal(memory, now, known=True, exclude=memory.recalled if memory else ())
    if signal is None:
        return None
    days = days_since_last(memory, now)
    angle = (rng or random).choice(opening_angles(signal, days))
    quote_key = dict(_QUOTE_ANGLES).get(angle)
    quote = getattr(signal, quote_key) if quote_key else None
    return OpeningHook(
        topic=signal.value,
        angle=angle,
        quote=quote.text if quote else None,
        days_since=days,
    )


def recheck_topic(
    memory: LongTermMemory | None,
    now: datetime,
    *,
    exclude: Iterable[str],
) -> str | None:
    """대화 중 한 번 다시 확인할 기억. 흐려지고 있지만 아직 잊지 않은 관심사다."""
    return _recallable(memory, now, known=False, exclude=exclude)
