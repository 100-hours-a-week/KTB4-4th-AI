from datetime import datetime
from typing import Literal

from pydantic import Field

from app.api.schemas.common import ApiModel, UserId
from app.api.schemas.profile import TasteField
from app.domain.profile.memory import (
    MAX_MEMORY_SIGNALS,
    MEMORY_SCHEMA_VERSION,
    RECALLED_LIMIT,
    LongTermMemory,
)
from app.domain.profile.merger import (
    MAX_AXES,
    link_role_for,
    normalize_text,
    visibility_for,
)
from app.domain.profile.models import (
    DeferralReason,
    EvidenceType,
    GroundedQuote,
    InsightKind,
    IntentType,
    PreferenceAspect,
    ProfileSignal,
    SignalStatus,
)
from app.domain.profile.models import TasteField as DomainTasteField

# 싫어하는 것·피해야 하는 것은 상한 없이 남기므로 저장 상한보다 넉넉히 받는다.
MAX_MEMORY_SIGNAL_INPUT = MAX_MEMORY_SIGNALS * 2


class MemoryQuote(ApiModel):
    text: str = Field(min_length=1, max_length=160)
    source_turn: int = Field(ge=0)
    conversation_id: int | None = Field(default=None, gt=0)

    def to_domain(self) -> GroundedQuote:
        return GroundedQuote(self.text, self.source_turn, self.conversation_id)

    @classmethod
    def from_domain(cls, quote: GroundedQuote | None) -> "MemoryQuote | None":
        return cls(**quote.to_dict()) if quote else None


class MemorySignal(ApiModel):
    field: TasteField
    value: str = Field(min_length=1, max_length=20)
    # 원래 확신도. 시간 감쇠는 AI 서버가 updatedAt을 보고 매번 다시 계산한다.
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1, max_length=40)
    evidence_type: Literal["explicit", "confirmed", "inferred"]
    mention_count: int = Field(ge=1)
    intent_type: Literal["need", "want", "both"] | None = None
    deferral_reason: Literal["justification", "price", "timing"] | None = None
    aspect: Literal["attribute", "sensory", "situation", "criterion", "motive"] | None = None
    target: str | None = Field(default=None, max_length=20)
    taxonomy_path: list[str] | None = None
    first_seen_at: datetime
    # 마지막으로 대화에서 확인한 시각
    updated_at: datetime
    insight_kind: InsightKind = InsightKind.PREFERENCE
    context: MemoryQuote | None = None
    reason: MemoryQuote | None = None
    origin: MemoryQuote | None = None

    def to_domain(self) -> ProfileSignal:
        field = DomainTasteField(self.field)
        return ProfileSignal(
            field=field,
            value=self.value,
            normalized_value=normalize_text(self.value),
            confidence=self.confidence,
            link_role=link_role_for(field),
            visibility=visibility_for(field),
            intent_type=IntentType(self.intent_type) if self.intent_type else None,
            deferral_reason=DeferralReason(self.deferral_reason) if self.deferral_reason else None,
            evidence=self.evidence,
            evidence_type=EvidenceType(self.evidence_type),
            first_seen_at=self.first_seen_at,
            updated_at=self.updated_at,
            mention_count=self.mention_count,
            status=SignalStatus.ACTIVE,
            aspect=PreferenceAspect(self.aspect) if self.aspect else None,
            target=self.target,
            taxonomy_path=tuple(self.taxonomy_path) if self.taxonomy_path else None,
            insight_kind=self.insight_kind,
            context=self.context.to_domain() if self.context else None,
            reason=self.reason.to_domain() if self.reason else None,
            origin=self.origin.to_domain() if self.origin else None,
        )

    @classmethod
    def from_domain(cls, signal: ProfileSignal) -> "MemorySignal":
        return cls(
            field=signal.field.value,
            value=signal.value,
            confidence=round(signal.confidence, 3),
            evidence=signal.evidence,
            evidence_type=signal.evidence_type.value,
            mention_count=signal.mention_count,
            intent_type=signal.intent_type.value if signal.intent_type else None,
            deferral_reason=signal.deferral_reason.value if signal.deferral_reason else None,
            aspect=signal.aspect.value if signal.aspect else None,
            target=signal.target,
            taxonomy_path=list(signal.taxonomy_path) if signal.taxonomy_path else None,
            first_seen_at=signal.first_seen_at,
            updated_at=signal.updated_at,
            insight_kind=signal.insight_kind,
            context=MemoryQuote.from_domain(signal.context),
            reason=MemoryQuote.from_domain(signal.reason),
            origin=MemoryQuote.from_domain(signal.origin),
        )


class ConversationMemory(ApiModel):
    """대화방 사이에 이어지는 장기 기억. Backend는 내용을 해석하지 않고 그대로 저장했다가 보낸다."""

    schema_version: Literal["1.0"]
    user_id: UserId
    session_count: int = Field(ge=1)
    updated_at: datetime
    signals: list[MemorySignal] = Field(max_length=MAX_MEMORY_SIGNAL_INPUT)
    axes: list[str] = Field(max_length=MAX_AXES)
    # 최근 대화에서 니쥬가 먼저 꺼낸 이야기. 다음 첫인사에서 같은 이야기를 반복하지 않는다.
    recalled: list[str] = Field(max_length=RECALLED_LIMIT)

    def to_domain(self) -> LongTermMemory:
        return LongTermMemory(
            user_id=self.user_id,
            signals=[signal.to_domain() for signal in self.signals],
            axes=list(self.axes),
            session_count=self.session_count,
            updated_at=self.updated_at,
            recalled=list(self.recalled),
        )

    @classmethod
    def from_domain(cls, memory: LongTermMemory) -> "ConversationMemory":
        return cls(
            schema_version=MEMORY_SCHEMA_VERSION,
            user_id=memory.user_id,
            session_count=memory.session_count,
            updated_at=memory.updated_at,
            signals=[MemorySignal.from_domain(signal) for signal in memory.signals],
            axes=list(memory.axes),
            recalled=list(memory.recalled),
        )
