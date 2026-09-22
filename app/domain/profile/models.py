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


@dataclass(slots=True, frozen=True)
class DropRef:
    field: TasteField
    value: str


@dataclass(slots=True, frozen=True)
class GoalAssessment:
    goal: str
    status: AssessmentStatus


@dataclass(slots=True, frozen=True)
class ExtractionDelta:
    items: tuple[ExtractedItem, ...] = ()
    axes: tuple[str, ...] = ()
    drop: tuple[DropRef, ...] = ()
    goal_assessment: GoalAssessment | None = None

    @classmethod
    def empty(cls, *, goal: str | None = None) -> ExtractionDelta:
        assessment = None
        if goal is not None:
            assessment = GoalAssessment(goal=goal, status=AssessmentStatus.UNRESOLVED)
        return cls(goal_assessment=assessment)


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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProfileSignal:
        intent = payload.get("intent_type")
        deferral = payload.get("deferral_reason")
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
