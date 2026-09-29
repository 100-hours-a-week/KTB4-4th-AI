from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class TasteField(StrEnum):
    INTERESTS = "interests"
    HOBBIES = "hobbies"
    PREFERENCES = "preferences"
    LIFESTYLE = "lifestyle"
    WANTS = "wants"
    UNAFFORDABLE = "unaffordable"
    CONSUMABLES = "consumables"
    OWNED = "owned"
    DISLIKES = "dislikes"
    CONSTRAINTS = "constraints"


class LinkRole(StrEnum):
    QUERY = "query"
    WEIGHT = "weight"
    FILTER = "filter"


class Visibility(StrEnum):
    PRIVATE = "private"
    FRIENDS = "friends"
    PUBLIC = "public"


class IntentType(StrEnum):
    NEED = "need"
    WANT = "want"
    BOTH = "both"


class DeferralReason(StrEnum):
    JUSTIFICATION = "justification"
    PRICE = "price"
    TIMING = "timing"


class EvidenceType(StrEnum):
    EXPLICIT = "explicit"
    CONFIRMED = "confirmed"
    INFERRED = "inferred"


class PreferenceAspect(StrEnum):
    """취향(preferences)이 대상의 어떤 결을 말하는지."""

    ATTRIBUTE = "attribute"  # 속성, 스타일
    SENSORY = "sensory"  # 맛, 소리, 촉감 같은 감각
    SITUATION = "situation"  # 선호하는 때와 상황, 분위기
    CRITERION = "criterion"  # 고를 때 중요하게 보는 기준
    MOTIVE = "motive"  # 그것을 선택하는 이유


class SignalStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUPERSEDED = "superseded"


class AssessmentStatus(StrEnum):
    FOUND = "found"
    CONFIRMED_NONE = "confirmed_none"
    UNRESOLVED = "unresolved"


@dataclass(slots=True, frozen=True)
class ExtractedItem:
    field: TasteField
    value: str
    confidence: float
    evidence: str
    evidence_type: EvidenceType
    intent_type: IntentType | None = None
    deferral_reason: DeferralReason | None = None
    aspect: PreferenceAspect | None = None
    target: str | None = None


@dataclass(slots=True, frozen=True)
class DropRef:
    field: TasteField
    value: str


@dataclass(slots=True, frozen=True)
class ExtractionDelta:
    items: tuple[ExtractedItem, ...] = ()
    axes: tuple[str, ...] = ()
    drop: tuple[DropRef, ...] = ()
    # 사용자가 직전 AI 질문에 해당하는 게 없다고 명확히 답했는지. 목표 판정은 코드가 한다.
    none_answer: bool = False


@dataclass(slots=True)
class ProfileSignal:
    field: TasteField
    value: str
    normalized_value: str
    confidence: float
    link_role: LinkRole
    visibility: Visibility
    intent_type: IntentType | None
    deferral_reason: DeferralReason | None
    evidence: str
    evidence_type: EvidenceType
    first_seen_at: datetime
    updated_at: datetime
    mention_count: int = 1
    status: SignalStatus = SignalStatus.ACTIVE
    source_turn: int = 0
    # preferences에만 쓴다. 어떤 결의 취향인지와, 어느 관심사에 붙은 취향인지.
    aspect: PreferenceAspect | None = None
    target: str | None = None

    @property
    def deferral_signal(self) -> bool:
        return self.deferral_reason is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "link_role": self.link_role.value,
            "visibility": self.visibility.value,
            "intent_type": self.intent_type.value if self.intent_type else None,
            "deferral_reason": self.deferral_reason.value if self.deferral_reason else None,
            "evidence": self.evidence,
            "evidence_type": self.evidence_type.value,
            "first_seen_at": self.first_seen_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "mention_count": self.mention_count,
            "status": self.status.value,
            "source_turn": self.source_turn,
            "aspect": self.aspect.value if self.aspect else None,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProfileSignal:
        intent = payload.get("intent_type")
        deferral = payload.get("deferral_reason")
        aspect = payload.get("aspect")
        return cls(
            field=TasteField(payload["field"]),
            value=payload["value"],
            normalized_value=payload["normalized_value"],
            confidence=float(payload["confidence"]),
            link_role=LinkRole(payload["link_role"]),
            visibility=Visibility(payload["visibility"]),
            intent_type=IntentType(intent) if intent else None,
            deferral_reason=DeferralReason(deferral) if deferral else None,
            evidence=payload["evidence"],
            evidence_type=EvidenceType(payload["evidence_type"]),
            first_seen_at=datetime.fromisoformat(payload["first_seen_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            mention_count=int(payload.get("mention_count", 1)),
            status=SignalStatus(payload.get("status", SignalStatus.ACTIVE)),
            source_turn=int(payload.get("source_turn", 0)),
            aspect=PreferenceAspect(aspect) if aspect else None,
            target=payload.get("target"),
        )


@dataclass(slots=True)
class ProfileState:
    signals: list[ProfileSignal] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)

    def active_signals(self) -> list[ProfileSignal]:
        return [signal for signal in self.signals if signal.status == SignalStatus.ACTIVE]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": [signal.to_dict() for signal in self.signals],
            "axes": list(self.axes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ProfileState:
        if not payload:
            return cls()
        return cls(
            signals=[ProfileSignal.from_dict(item) for item in payload.get("signals", [])],
            axes=list(payload.get("axes", [])),
        )


def utc_now() -> datetime:
    return datetime.now(UTC)
