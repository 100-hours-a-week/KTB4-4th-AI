from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from app.application.conversation_prompts import (
    build_candidate_messages,
    build_extraction_messages,
    build_opening_messages,
    build_reply_messages,
)
from app.application.ports.judgment_gateway import JudgmentGateway
from app.application.ports.model_gateway import Message, ModelGateway
from app.application.taste_judgment import (
    Candidate,
    CandidateDeltaPayload,
    build_judgment_request,
    correction_targets,
    judgment_delta,
    parse_candidates,
    remention_candidates,
    without_frequency_variants,
)
from app.domain.conversation.guards import closing_reply, sanitize_response
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationMove,
    ConversationState,
    ConversationStyle,
    ConversationTurn,
    GoalDecision,
    ReadinessResult,
    SessionStatus,
)
from app.domain.conversation.policy import (
    MAX_TURNS,
    advance_thread,
    apply_goal_assessment,
    conversation_progress,
    decide_goal,
    fields_for_goal,
    known_query_values,
    recommendation_readiness,
    record_companion_turn,
    record_goal_attempt,
    record_interest_progress,
    reflection_clues,
)
from app.domain.profile.merger import MergeResult, ProfileMerger, normalize_text
from app.domain.profile.models import (
    AssessmentStatus,
    DeferralReason,
    DropRef,
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
    IntentType,
    LinkRole,
    PreferenceAspect,
    TasteField,
    utc_now,
)

logger = logging.getLogger(__name__)

MAX_EXTRACTED_ITEMS = 5
MAX_VALUE_LENGTH = 20
MAX_EVIDENCE_LENGTH = 40
EXTRACTION_CONTEXT_USER_TURNS = 4


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
    aspect: PreferenceAspect | None = None
    target: ShortValue | None = None

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


class ExtractionDeltaPayload(_ExtractionModel):
    """모델에 안내하는 출력 스키마. 실제 파싱은 parse_extraction이 항목 단위로 한다."""

    items: list[ExtractedItemPayload] = Field(default_factory=list, max_length=MAX_EXTRACTED_ITEMS)
    axes: list[AxisValue] = Field(default_factory=list, max_length=3)
    drop: list[DropRefPayload] = Field(default_factory=list, max_length=3)
    none_answer: bool = False


def _repair_item(raw: Mapping[str, object]) -> dict[str, object]:
    item = dict(raw)
    evidence = item.get("evidence")
    if isinstance(evidence, str):
        item["evidence"] = evidence.strip()[:MAX_EVIDENCE_LENGTH]
    has_reason = item.get("deferralReason") is not None
    if item.get("field") == TasteField.UNAFFORDABLE.value and not has_reason:
        item["field"] = TasteField.WANTS.value
    elif item.get("field") != TasteField.UNAFFORDABLE.value and has_reason:
        item.pop("deferralReason", None)
    # aspect와 target은 취향에만 의미가 있다. 다른 field에 붙어 오면 항목은 살리고 떼어 낸다.
    if item.get("field") != TasteField.PREFERENCES.value:
        item.pop("aspect", None)
        item.pop("target", None)
    elif isinstance(item.get("target"), str) and not item["target"].strip():
        item["target"] = None
    return item


def _to_item(payload: ExtractedItemPayload) -> ExtractedItem:
    return ExtractedItem(
        field=payload.field,
        value=payload.value,
        confidence=payload.confidence,
        evidence=payload.evidence,
        evidence_type=payload.evidence_type,
        intent_type=payload.intent_type,
        deferral_reason=payload.deferral_reason,
        aspect=payload.aspect,
        target=payload.target,
    )


def parse_extraction(raw: object) -> tuple[ExtractionDelta, list[str]]:
    """추출 결과를 항목 단위로 검증한다. 틀린 항목만 버리고 나머지는 살린다.

    최상위가 객체가 아닐 때만 예외를 던져 재시도하게 한다.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("extraction result is not an object")

    dropped: list[str] = []
    items: list[ExtractedItem] = []
    raw_items = raw.get("items")
    for raw_item in raw_items if isinstance(raw_items, list) else []:
        if len(items) >= MAX_EXTRACTED_ITEMS:
            dropped.append("item limit exceeded")
            break
        if not isinstance(raw_item, Mapping):
            dropped.append("item is not an object")
            continue
        try:
            items.append(_to_item(ExtractedItemPayload.model_validate(_repair_item(raw_item))))
        except ValidationError as error:
            dropped.append(f"{raw_item.get('value')!r}: {error.errors()[0]['msg']}")

    axes: list[str] = []
    raw_axes = raw.get("axes")
    for axis in raw_axes if isinstance(raw_axes, list) else []:
        if isinstance(axis, str) and 0 < len(axis.strip()) <= MAX_EVIDENCE_LENGTH:
            axes.append(axis.strip())

    drops: list[DropRef] = []
    raw_drops = raw.get("drop")
    for raw_drop in raw_drops if isinstance(raw_drops, list) else []:
        try:
            ref = DropRefPayload.model_validate(raw_drop)
        except ValidationError:
            dropped.append(f"drop {raw_drop!r}")
            continue
        drops.append(DropRef(field=ref.field, value=ref.value))

    delta = ExtractionDelta(
        items=tuple(items),
        axes=tuple(axes[:3]),
        drop=tuple(drops[:3]),
        none_answer=raw.get("noneAnswer") is True,
    )
    return delta, dropped


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
        judgment_gateway: JudgmentGateway | None = None,
        judgment_model: str | None = None,
        judgment_timeout_seconds: float = 10.0,
        conversation_style: ConversationStyle = ConversationStyle.EXPLORE,
    ) -> None:
        self._model_gateway = model_gateway
        self._extraction_model = extraction_model
        self._merger = merger or ProfileMerger()
        self._extraction_retries = extraction_retries
        self._extraction_timeout_seconds = extraction_timeout_seconds
        self._judgment_gateway = judgment_gateway
        self._judgment_model = judgment_model
        self._judgment_timeout_seconds = judgment_timeout_seconds
        self._conversation_style = conversation_style

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
            conversation_style=self._conversation_style,
        )
        return PreparedSession(
            state=state,
            messages=tuple(
                build_opening_messages(now=resolved_now, style=self._conversation_style)
            ),
        )

    def record_opening(
        self,
        state: ConversationState,
        raw_greeting: str,
        *,
        now: datetime | None = None,
    ) -> str:
        greeting = self.guard_reply(
            raw_greeting, ConversationGoal.OPENING, style=state.conversation_style
        )
        resolved_now = now or utc_now()
        state.history.append(
            ConversationTurn(role="assistant", content=greeting, created_at=resolved_now)
        )
        state.last_active_at = resolved_now
        return greeting

    def prepare_turn(
        self,
        state: ConversationState,
        utterance: str,
        *,
        goal: ConversationGoal | None = None,
    ) -> PreparedTurn:
        decision = GoalDecision(goal) if goal is not None else decide_goal(state, utterance)
        messages = build_reply_messages(
            state, decision.goal, utterance, move=decision.move, scene=decision.scene
        )
        return PreparedTurn(decision=decision, messages=tuple(messages))

    def guard_reply(
        self,
        reply: str,
        goal: ConversationGoal,
        *,
        style: ConversationStyle = ConversationStyle.EXPLORE,
    ) -> str:
        return sanitize_response(
            reply,
            goal,
            concise=style == ConversationStyle.REFLECTIVE,
            companion=style == ConversationStyle.COMPANION,
        )

    async def extract(
        self,
        state: ConversationState,
        utterance: str,
    ) -> tuple[ExtractionDelta, bool]:
        """이번 발화에서 취향을 추출한다. 응답 결과를 쓰지 않아 응답 생성과 동시에 돌릴 수 있다."""
        return await self._extract(state, utterance)

    async def _extract(
        self,
        state: ConversationState,
        utterance: str,
    ) -> tuple[ExtractionDelta, bool]:
        if self._judgment_gateway is not None and self._judgment_model is not None:
            return await self._extract_with_judgment(
                state,
                utterance,
                judgment_gateway=self._judgment_gateway,
                judgment_model=self._judgment_model,
            )
        messages = build_extraction_messages(state, utterance)
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
                        delta, dropped = parse_extraction(raw)
                        if dropped:
                            logger.info("extraction items dropped: %s", dropped)
                        return delta, False
                    except Exception:
                        # The user-facing reply has already completed. Extraction failures are
                        # retried and then downgraded to an empty delta so they never break chat.
                        continue
        except TimeoutError:
            pass
        return ExtractionDelta(), True

    async def _extract_candidates(
        self,
        state: ConversationState,
        utterance: str,
    ) -> tuple[list[Candidate], list[str]] | None:
        messages = build_candidate_messages(state, utterance)
        schema = CandidateDeltaPayload.model_json_schema()
        try:
            async with asyncio.timeout(self._extraction_timeout_seconds):
                for _ in range(self._extraction_retries + 1):
                    try:
                        raw = await self._model_gateway.structured(
                            messages,
                            model=self._extraction_model,
                            json_schema=schema,
                        )
                        candidates, axes, dropped = parse_candidates(raw)
                        if dropped:
                            logger.info("candidates dropped: %s", dropped)
                        return candidates, axes
                    except Exception:
                        continue
        except TimeoutError:
            pass
        return None

    async def _extract_with_judgment(
        self,
        state: ConversationState,
        utterance: str,
        *,
        judgment_gateway: JudgmentGateway,
        judgment_model: str,
    ) -> tuple[ExtractionDelta, bool]:
        """추출 모델로 언급된 대상을 찾고, 대상마다 사용자의 태도는 판단 모델이 정한다."""
        extracted = await self._extract_candidates(state, utterance)
        if extracted is None:
            return ExtractionDelta(), True
        candidates, axes = extracted
        candidates = without_frequency_variants(
            candidates,
            known={signal.normalized_value for signal in state.profile.stored_signals()},
        )
        candidates.extend(remention_candidates(state.profile, utterance, candidates))
        request = build_judgment_request(
            state,
            utterance,
            candidates,
            correction_targets(state, utterance),
        )
        if not request.questions:
            return ExtractionDelta(axes=tuple(axes)), False
        body = None
        try:
            async with asyncio.timeout(self._judgment_timeout_seconds):
                for _ in range(self._extraction_retries + 1):
                    try:
                        body = await judgment_gateway.decide(
                            request.state,
                            request.questions,
                            model=judgment_model,
                        )
                        break
                    except Exception:
                        continue
        except TimeoutError:
            pass
        if body is None:
            return ExtractionDelta(), True
        delta, judgments = judgment_delta(
            candidates,
            body["answers"],
            request.sentences,
            axes=axes,
            targets=request.targets,
        )
        logger.info("taste judgment (%s):", body.get("model"))
        for drop in delta.drop:
            logger.info("  정정: %s (%s) 내림", drop.value, drop.field.value)
        for judgment in judgments:
            logger.info(
                "  %s | %s %.2f | %s | %s → %s",
                judgment.candidate.subject,
                judgment.stance.value,
                judgment.confidence,
                judgment.material.value if judgment.material else "-",
                "/".join(judgment.taxonomy_path) if judgment.taxonomy_path else "-",
                judgment.field.value if judgment.field else "저장 안 함",
            )
        return delta, False

    async def complete_turn(
        self,
        state: ConversationState,
        *,
        utterance: str,
        raw_reply: str,
        goal: ConversationGoal,
        completion_reason: CompletionReason | None = None,
        now: datetime | None = None,
        extraction: tuple[ExtractionDelta, bool] | None = None,
        move: ConversationMove | None = None,
        scene: str | None = None,
    ) -> CompletedTurn:
        resolved_now = now or utc_now()
        resolved_completion_reason = completion_reason
        if goal == ConversationGoal.WRAP and resolved_completion_reason is None:
            resolved_completion_reason = decide_goal(state, utterance).completion_reason
        reply = self.guard_reply(raw_reply, goal, style=state.conversation_style)
        # 되비추기에 쓴 단서. 이번 턴 병합 전 상태로 프롬프트와 같은 것을 고른다.
        reflecting = goal == ConversationGoal.REFLECT or move == ConversationMove.REFLECT_BACK
        clues = reflection_clues(state) if reflecting else []
        if extraction is None:
            extraction = await self._extract(state, utterance)
        delta, extraction_failed = extraction
        known_values = known_query_values(state)
        merge_result = self._merger.merge(
            state.profile,
            delta,
            utterance=utterance,
            now=resolved_now,
            source_turn=state.turn_count + 1,
            context_utterances=state.user_turns()[-EXTRACTION_CONTEXT_USER_TURNS:],
        )
        state.profile = merge_result.profile
        record_interest_progress(state, known_values, merge_result.accepted)
        record_goal_attempt(state, goal)
        apply_goal_assessment(state, goal, _assess_goal(goal, delta, merge_result))
        new_interests = [
            signal.value
            for signal in merge_result.accepted
            if signal.link_role == LinkRole.QUERY
            and normalize_text(signal.value) not in known_values
        ]
        advance_thread(
            state,
            goal,
            new_interests=new_interests,
            answer_depth=delta.answer_depth,
        )
        # history에 이번 발화를 넣기 전에 기록해야 평균 길이와 비교된다.
        record_companion_turn(
            state,
            utterance=utterance,
            reply=reply,
            move=move,
            scene=scene,
            answer_depth=delta.answer_depth,
            new_interests=new_interests,
        )
        if reflecting:
            state.reflection_values = [signal.value for signal in clues]

        state.history.append(
            ConversationTurn(role="user", content=utterance, created_at=resolved_now)
        )
        state.history.append(
            ConversationTurn(role="assistant", content=reply, created_at=resolved_now)
        )
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

        # 마무리 차례인데 모델이 질문으로 끝냈으면 질문을 떼고 종료 멘트를 붙인다.
        if goal == ConversationGoal.WRAP and reply.endswith("?"):
            reply = closing_reply(reply)
            state.history[-1] = ConversationTurn(
                role="assistant", content=reply, created_at=resolved_now
            )
        # 되비추기 질문을 방금 했으면 답을 들어야 하므로 이번 턴에는 잠그지 않는다.
        # 최대 턴에 도달했으면 더 받을 수 없으므로 되비추기 차례였어도 잠근다.
        reached_max_turns = state.turn_count >= MAX_TURNS
        if next_decision.goal == ConversationGoal.WRAP and (
            goal != ConversationGoal.REFLECT or reached_max_turns
        ):
            state.status = SessionStatus.INPUT_LOCKED
            state.completion_reason = next_decision.completion_reason
            # 종료가 판정되면 최대 턴에 도달했어도 방금 만든 질문 대신 종료 멘트를 보낸다.
            if goal != ConversationGoal.WRAP:
                reply = closing_reply(reply)
                state.history[-1] = ConversationTurn(
                    role="assistant", content=reply, created_at=resolved_now
                )

        if state.conversation_style == ConversationStyle.COMPANION:
            state.progress_floor = conversation_progress(state)

        return CompletedTurn(
            reply=reply,
            state=state,
            merge_result=merge_result,
            readiness=readiness,
            next_decision=next_decision,
            extraction_failed=extraction_failed,
        )


def _assess_goal(
    goal: ConversationGoal,
    delta: ExtractionDelta,
    merge_result: MergeResult,
) -> AssessmentStatus:
    """이번 턴에 목표 영역의 신호가 들어왔는지로 목표 달성 여부를 판단한다."""
    if any(signal.field in fields_for_goal(goal) for signal in merge_result.accepted):
        return AssessmentStatus.FOUND
    if delta.none_answer:
        return AssessmentStatus.CONFIRMED_NONE
    return AssessmentStatus.UNRESOLVED
