from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.application.conversation_prompts import (
    build_extraction_messages,
    build_opening_messages,
    build_reply_messages,
)
from app.application.ports.model_gateway import Message, ModelGateway
from app.domain.conversation.guards import sanitize_response
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    ConversationTurn,
    GoalDecision,
    ReadinessResult,
    SessionStatus,
)
from app.domain.conversation.policy import (
    apply_goal_assessment,
    decide_goal,
    fields_for_goal,
    recommendation_readiness,
    record_goal_attempt,
)
from app.domain.profile.merger import MergeResult, ProfileMerger
from app.domain.profile.models import (
    AssessmentStatus,
    DeferralReason,
    DropRef,
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
    GoalAssessment,
    IntentType,
    TasteField,
    utc_now,
)


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(word.capitalize() for word in rest)


ShortValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20),
]
EvidenceValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=40),
]
AxisValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=40),
]


class _ExtractionModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ExtractedItemPayload(_ExtractionModel):
    field: TasteField
    value: ShortValue
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: EvidenceValue
    evidence_type: EvidenceType
    intent_type: IntentType | None = None
    deferral_reason: DeferralReason | None = None

    @model_validator(mode="after")
    def validate_deferral(self) -> Self:
        if self.field == TasteField.UNAFFORDABLE and self.deferral_reason is None:
            raise ValueError("unaffordable requires deferralReason")
        if self.field != TasteField.UNAFFORDABLE and self.deferral_reason is not None:
            raise ValueError("deferralReason is only valid for unaffordable")
        return self


class DropRefPayload(_ExtractionModel):
    field: TasteField
    value: ShortValue


class GoalAssessmentPayload(_ExtractionModel):
    goal: ConversationGoal
    status: AssessmentStatus


class ExtractionDeltaPayload(_ExtractionModel):
    items: list[ExtractedItemPayload] = Field(default_factory=list, max_length=3)
    axes: list[AxisValue] = Field(default_factory=list, max_length=3)
    drop: list[DropRefPayload] = Field(default_factory=list, max_length=3)
    goal_assessment: GoalAssessmentPayload

    def to_domain(self) -> ExtractionDelta:
        return ExtractionDelta(
            items=tuple(
                ExtractedItem(
                    field=item.field,
                    value=item.value,
                    confidence=item.confidence,
                    evidence=item.evidence,
                    evidence_type=item.evidence_type,
                    intent_type=item.intent_type,
                    deferral_reason=item.deferral_reason,
                )
                for item in self.items
            ),
            axes=tuple(self.axes),
            drop=tuple(DropRef(field=item.field, value=item.value) for item in self.drop),
            goal_assessment=GoalAssessment(
                goal=self.goal_assessment.goal.value,
                status=self.goal_assessment.status,
            ),
        )


@dataclass(slots=True, frozen=True)
class PreparedTurn:
    decision: GoalDecision
    messages: tuple[Message, ...]


@dataclass(slots=True, frozen=True)
class PreparedSession:
    state: ConversationState
    messages: tuple[Message, ...]


@dataclass(slots=True, frozen=True)
class CompletedTurn:
    reply: str
    state: ConversationState
    merge_result: MergeResult
    readiness: ReadinessResult
    next_decision: GoalDecision
    extraction_failed: bool


class ConversationService:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        extraction_model: str,
        merger: ProfileMerger | None = None,
        extraction_retries: int = 2,
        extraction_timeout_seconds: float = 30.0,
    ) -> None:
        self._model_gateway = model_gateway
        self._extraction_model = extraction_model
        self._merger = merger or ProfileMerger()
        self._extraction_retries = extraction_retries
        self._extraction_timeout_seconds = extraction_timeout_seconds

    def prepare_session(
        self,
        *,
        user_id: int,
        conversation_room_id: int,
        now: datetime | None = None,
    ) -> PreparedSession:
        invalid_user_id = isinstance(user_id, bool) or user_id <= 0
        invalid_room_id = isinstance(conversation_room_id, bool) or conversation_room_id <= 0
        if invalid_user_id or invalid_room_id:
            raise ValueError("user_id and conversation_room_id are required")
        resolved_now = now or utc_now()
        state = ConversationState(
            user_id=user_id,
            conversation_room_id=conversation_room_id,
            created_at=resolved_now,
            last_active_at=resolved_now,
        )
        return PreparedSession(state=state, messages=tuple(build_opening_messages()))

    def record_opening(
        self,
        state: ConversationState,
        raw_greeting: str,
        *,
        now: datetime | None = None,
    ) -> str:
        greeting = self.guard_reply(raw_greeting, ConversationGoal.OPENING)
        state.history.append(ConversationTurn(role="assistant", content=greeting))
        state.last_active_at = now or utc_now()
        return greeting

    def prepare_turn(
        self,
        state: ConversationState,
        utterance: str,
        *,
        goal: ConversationGoal | None = None,
    ) -> PreparedTurn:
        decision = GoalDecision(goal) if goal is not None else decide_goal(state, utterance)
        messages = build_reply_messages(state, decision.goal, utterance)
        return PreparedTurn(decision=decision, messages=tuple(messages))

    def guard_reply(self, reply: str, goal: ConversationGoal) -> str:
        return sanitize_response(reply, goal)

    async def _extract(
        self,
        state: ConversationState,
        goal: ConversationGoal,
        assistant_reply: str,
        utterance: str,
    ) -> tuple[ExtractionDelta, bool]:
        messages = build_extraction_messages(state, goal, assistant_reply, utterance)
        schema = ExtractionDeltaPayload.model_json_schema(by_alias=True)
        try:
            async with asyncio.timeout(self._extraction_timeout_seconds):
                for _ in range(self._extraction_retries + 1):
                    try:
                        raw = await self._model_gateway.structured(
                            messages,
                            model=self._extraction_model,
                            json_schema=schema,
                        )
                        parsed = ExtractionDeltaPayload.model_validate(raw)
                        delta = parsed.to_domain()
                        if (
                            delta.goal_assessment is None
                            or delta.goal_assessment.goal != goal.value
                        ):
                            raise ValueError("goalAssessment does not match the current goal")
                        return delta, False
                    except Exception:
                        # The user-facing reply has already completed. Extraction failures are
                        # retried and then downgraded to an empty delta so they never break chat.
                        continue
        except TimeoutError:
            pass
        return ExtractionDelta.empty(goal=goal.value), True

    async def complete_turn(
        self,
        state: ConversationState,
        *,
        utterance: str,
        raw_reply: str,
        goal: ConversationGoal,
        completion_reason: CompletionReason | None = None,
        now: datetime | None = None,
    ) -> CompletedTurn:
        resolved_now = now or utc_now()
        resolved_completion_reason = completion_reason
        if goal == ConversationGoal.WRAP and resolved_completion_reason is None:
            resolved_completion_reason = decide_goal(state, utterance).completion_reason
        reply = self.guard_reply(raw_reply, goal)
        delta, extraction_failed = await self._extract(
            state,
            goal,
            reply,
            utterance,
        )
        merge_result = self._merger.merge(
            state.profile,
            delta,
            utterance=utterance,
            now=resolved_now,
            source_turn=state.turn_count + 1,
        )
        state.profile = merge_result.profile
        record_goal_attempt(state, goal)

        assessment = delta.goal_assessment
        if assessment is None:
            assessment_status = AssessmentStatus.UNRESOLVED
        elif assessment.status == AssessmentStatus.FOUND and not any(
            signal.field in fields_for_goal(goal) for signal in merge_result.accepted
        ):
            assessment_status = AssessmentStatus.UNRESOLVED
        else:
            assessment_status = assessment.status
        apply_goal_assessment(state, goal, assessment_status)

        state.history.append(ConversationTurn(role="user", content=utterance))
        state.history.append(ConversationTurn(role="assistant", content=reply))
        state.turn_count += 1
        state.last_active_at = resolved_now
        state.last_turn_extraction_failed = extraction_failed
        readiness = recommendation_readiness(state)
        if goal == ConversationGoal.WRAP:
            next_decision = GoalDecision(
                ConversationGoal.WRAP,
                resolved_completion_reason or CompletionReason.USER_EXIT,
            )
        else:
            next_decision = decide_goal(state, "")

        if next_decision.goal == ConversationGoal.WRAP:
            state.status = SessionStatus.INPUT_LOCKED
            state.completion_reason = next_decision.completion_reason

        return CompletedTurn(
            reply=reply,
            state=state,
            merge_result=merge_result,
            readiness=readiness,
            next_decision=next_decision,
            extraction_failed=extraction_failed,
        )
