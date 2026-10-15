import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, Path, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_use_cases, get_recommendation_service
from app.api.schemas.chat import (
    AnalyzeChatSessionResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatMessageStreamRequest,
    ChatStreamDelta,
    ChatStreamError,
    CloseChatSessionRequest,
    CloseChatSessionResponse,
    CreateChatSessionRequest,
    CreateChatSessionResponse,
    UpdateChatAnalysisRequest,
)
from app.api.schemas.common import JAVA_LONG_MAX, ApiModel, to_camel
from app.api.schemas.profile import ProfileItem, ScoredProfileKeywords, TasteProfile
from app.api.schemas.recommendation import RecommendationLists
from app.application.chat_use_cases import (
    AnalysisAlreadyCorrectedError,
    AnalysisKeywordsCanOnlyBeDeletedError,
    AnalysisOutdatedError,
    AnalysisRequiredError,
    ChatUseCases,
    ConfirmedTurn,
    ProfileAnalysis,
    ProfileTooSparseError,
    ReplyDelta,
    ResponseModelUnavailableError,
    SessionClosedError,
    SessionExistsError,
    SessionInputLockedError,
    SessionNotFoundError,
    TurnInProgressError,
)
from app.application.conversation_service import CompletedTurn
from app.application.conversation_state_store import SessionLifetimeExceededError
from app.application.ports.recommendation_service import RecommendationService
from app.core.errors import ApiError
from app.domain.conversation.models import SessionStatus
from app.domain.conversation.policy import (
    MAX_TURNS,
    conversation_progress,
    recommendation_readiness,
)
from app.domain.profile.merger import taste_rank_score
from app.domain.profile.models import ProfileSignal, TasteField

logger = logging.getLogger(__name__)

router = APIRouter()

# 보낼 이벤트가 없을 때 연결이 살아 있음을 알리는 주석 간격. BE는 30초 동안 아무것도 없으면 끊는다.
STREAM_PING_INTERVAL_SECONDS = 10.0
# 연결이 끊긴 뒤 생성 작업이 취소되고 잠금이 풀리기를 기다리는 최대 시간
STREAM_CANCEL_WAIT_SECONDS = 2.0
_STREAM_END = object()

ConversationRoomPath = Annotated[
    int,
    Path(alias="conversationRoomId", ge=1, le=JAVA_LONG_MAX),
]


def _raise_chat_error(error: Exception) -> None:
    if isinstance(error, (SessionNotFoundError, SessionLifetimeExceededError)):
        raise ApiError(
            status_code=404,
            code="SESSION_NOT_FOUND",
            message="세션이 없거나 만료되었습니다.",
        ) from error
    if isinstance(error, SessionExistsError):
        raise ApiError(
            status_code=409,
            code="SESSION_EXISTS",
            message="이미 존재하는 대화 세션입니다.",
        ) from error
    if isinstance(error, SessionClosedError):
        raise ApiError(
            status_code=409,
            code="SESSION_CLOSED",
            message="이미 종료된 대화 세션입니다.",
        ) from error
    if isinstance(error, SessionInputLockedError):
        raise ApiError(
            status_code=409,
            code="INPUT_LOCKED",
            message=(
                "현재 상태에서는 대화 메시지를 보낼 수 없습니다. "
                "취향 분석 결과를 확인하거나 수정해 주세요."
            ),
        ) from error
    if isinstance(error, TurnInProgressError):
        raise ApiError(
            status_code=409,
            code="TURN_IN_PROGRESS",
            message="이전 메시지를 아직 처리하고 있습니다.",
            retryable=True,
        ) from error
    if isinstance(error, ProfileTooSparseError):
        raise ApiError(
            status_code=422,
            code="PROFILE_TOO_SPARSE",
            message="취향 프로필을 만들기 위한 정보가 부족합니다.",
        ) from error
    if isinstance(error, AnalysisRequiredError):
        raise ApiError(
            status_code=409,
            code="ANALYSIS_REQUIRED",
            message="세션을 종료하기 전에 취향 분석 결과를 확인해 주세요.",
        ) from error
    if isinstance(error, AnalysisOutdatedError):
        raise ApiError(
            status_code=409,
            code="ANALYSIS_OUTDATED",
            message="대화가 변경되었습니다. 취향 분석 결과를 다시 확인해 주세요.",
        ) from error
    if isinstance(error, AnalysisAlreadyCorrectedError):
        raise ApiError(
            status_code=409,
            code="ANALYSIS_ALREADY_CORRECTED",
            message="취향 분석 결과는 한 번만 수정할 수 있습니다.",
        ) from error
    if isinstance(error, AnalysisKeywordsCanOnlyBeDeletedError):
        raise ApiError(
            status_code=422,
            code="KEYWORDS_DELETE_ONLY",
            message="키워드는 기존 분석 결과에서 삭제만 할 수 있습니다.",
        ) from error
    if isinstance(error, ResponseModelUnavailableError):
        raise ApiError(
            status_code=502,
            code="LLM_UNAVAILABLE",
            message="대화 모델을 호출할 수 없습니다.",
            retryable=True,
        ) from error
    raise error


def _grouped_items(signals: Iterable[ProfileSignal]) -> dict[TasteField, list[ProfileItem]]:
    grouped: dict[TasteField, list[ProfileItem]] = {field: [] for field in TasteField}
    for signal in signals:
        grouped[signal.field].append(
            ProfileItem(
                value=signal.value,
                confidence=signal.confidence,
                rank_score=taste_rank_score(signal),
                link_role=signal.link_role.value,
                visibility=signal.visibility.value,
                intent_type=signal.intent_type.value if signal.intent_type else None,
                deferral_signal=signal.deferral_signal,
                deferral_reason=(signal.deferral_reason.value if signal.deferral_reason else None),
                evidence=signal.evidence,
                aspect=signal.aspect.value if signal.aspect else None,
                target=signal.target,
                taxonomy_path=list(signal.taxonomy_path) if signal.taxonomy_path else None,
                first_seen_at=signal.first_seen_at,
                updated_at=signal.updated_at,
            )
        )
    return grouped


def _profile_response(result: ProfileAnalysis) -> TasteProfile:
    grouped = _grouped_items(result.profile.active_signals())
    return TasteProfile(
        schema_version="3.0",
        user_id=result.state.user_id,
        summary=result.summary,
        interests=grouped[TasteField.INTERESTS],
        hobbies=grouped[TasteField.HOBBIES],
        preferences=grouped[TasteField.PREFERENCES],
        lifestyle=grouped[TasteField.LIFESTYLE],
        wants=grouped[TasteField.WANTS],
        unaffordable=grouped[TasteField.UNAFFORDABLE],
        consumables=grouped[TasteField.CONSUMABLES],
        owned=grouped[TasteField.OWNED],
        dislikes=grouped[TasteField.DISLIKES],
        constraints=grouped[TasteField.CONSTRAINTS],
        axes=result.profile.axes,
    )


def recommendation_profile(result: ProfileAnalysis) -> dict[str, Any]:
    """추천 엔진에 넘기는 프로필. 백엔드로 보내는 프로필과 모양은 같지만 노출 한도 없이
    저장된 항목 전체를 담는다. 추천 쿼리는 뽑은 것 전체를 조합해야 목록이 다양해진다."""
    grouped = _grouped_items(result.profile.stored_signals())
    return {
        "schemaVersion": "3.0",
        "userId": result.state.user_id,
        "summary": result.summary,
        **{
            to_camel(field.value): [
                item.model_dump(mode="json", by_alias=True) for item in grouped[field]
            ]
            for field in TasteField
        },
        "axes": list(result.profile.axes),
    }


def _scored_keywords(analysis: ProfileAnalysis) -> ScoredProfileKeywords:
    def scored(keywords: tuple[str, ...]) -> list[dict[str, object]]:
        # 분석 이전 버전에서 저장된 세션에는 점수가 없으므로 0점으로 내려준다.
        return [
            {"value": keyword, "score": analysis.keyword_scores.get(keyword, 0.0)}
            for keyword in keywords
        ]

    return ScoredProfileKeywords.model_validate(
        {
            "taste": scored(analysis.taste_keywords),
            "interest": scored(analysis.interest_keywords),
        }
    )


def _analysis_response(analysis: ProfileAnalysis) -> AnalyzeChatSessionResponse:
    return AnalyzeChatSessionResponse(
        profile={
            "userId": analysis.state.user_id,
            "summary": analysis.summary,
            "keywords": _scored_keywords(analysis),
            "correctionAvailable": not analysis.state.analysis_patch_used,
        },
    )


@router.post(
    "/sessions",
    response_model=CreateChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_session(
    body: CreateChatSessionRequest,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> CreateChatSessionResponse:
    try:
        started = await use_cases.create_session(
            user_id=body.user_id,
            conversation_room_id=body.conversation_room_id,
        )
    except Exception as error:
        _raise_chat_error(error)
    return CreateChatSessionResponse(
        conversation_room_id=started.state.conversation_room_id,
        greeting=started.greeting,
        created_at=started.state.history[-1].created_at,
        expiration_at=started.state.expiration_at,
        max_turns=MAX_TURNS,
    )


@router.post(
    "/sessions/{conversationRoomId}/messages",
    response_model=ChatMessageResponse,
)
async def create_chat_message(
    conversation_room_id: ConversationRoomPath,
    body: ChatMessageRequest,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> ChatMessageResponse:
    try:
        completed = await use_cases.send_message(
            conversation_room_id,
            body.message,
            user_id=body.user_id,
        )
    except Exception as error:
        _raise_chat_error(error)
    return _completed_response(completed)


def _completed_response(completed: CompletedTurn) -> ChatMessageResponse:
    return ChatMessageResponse(
        reply=completed.reply,
        created_at=completed.state.history[-1].created_at,
        expiration_at=completed.state.expiration_at,
        turn=completed.state.turn_count,
        max_turns=MAX_TURNS,
        progress=conversation_progress(completed.state),
        can_close=completed.readiness.sufficient,
        input_locked=completed.state.status == SessionStatus.INPUT_LOCKED,
    )


def _confirmed_response(confirmed: ConfirmedTurn) -> ChatMessageResponse:
    """이미 확정한 턴을 다시 요청받았을 때 저장된 상태로 같은 응답을 만든다."""
    state = confirmed.state
    last_reply = state.history[-1]
    return ChatMessageResponse(
        reply=last_reply.content,
        created_at=last_reply.created_at,
        expiration_at=state.expiration_at,
        turn=state.turn_count,
        max_turns=MAX_TURNS,
        progress=conversation_progress(state),
        can_close=recommendation_readiness(state).sufficient,
        input_locked=state.status == SessionStatus.INPUT_LOCKED,
    )


def _sse_event(event: str, data: ApiModel) -> str:
    return f"event: {event}\ndata: {data.model_dump_json(by_alias=True)}\n\n"


def _stream_error(error: Exception) -> ChatStreamError:
    """스트림 도중 예외를 HTTP 오류와 같은 code·retryable로 바꾼다."""
    try:
        _raise_chat_error(error)
    except ApiError as api_error:
        return ChatStreamError(
            code=api_error.code,
            message=api_error.message,
            retryable=api_error.retryable,
        )
    except Exception:
        logger.exception("chat stream failed unexpectedly")
    return ChatStreamError(
        code="INTERNAL_ERROR",
        message="응답을 만들지 못했습니다.",
        retryable=True,
    )


async def _single(item: ConfirmedTurn) -> AsyncIterator[ConfirmedTurn]:
    yield item


async def _sse_stream(
    events: AsyncIterator[ReplyDelta | CompletedTurn | ConfirmedTurn],
) -> AsyncIterator[str]:
    """턴 생성은 별도 작업에서 돌리고, 이벤트가 뜸하면 주석(: ping)을 보낸다.

    호출한 쪽이 연결을 끊으면 이 제너레이터가 닫히며 생성 작업을 취소한다. 취소된 턴은
    저장하지 않는다.
    """
    queue: asyncio.Queue[object] = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in events:
                await queue.put(event)
        except Exception as error:
            await queue.put(_stream_error(error))
        finally:
            queue.put_nowait(_STREAM_END)

    producer = asyncio.create_task(produce())
    next_index = 0
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=STREAM_PING_INTERVAL_SECONDS)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if item is _STREAM_END:
                return
            if isinstance(item, ReplyDelta):
                yield _sse_event("delta", ChatStreamDelta(index=next_index, text=item.text))
                next_index += 1
            elif isinstance(item, CompletedTurn):
                yield _sse_event("completed", _completed_response(item))
            elif isinstance(item, ConfirmedTurn):
                yield _sse_event("completed", _confirmed_response(item))
            elif isinstance(item, ChatStreamError):
                yield _sse_event("error", item)
    finally:
        producer.cancel()
        # 바로 같은 메시지로 다시 요청해도 잠금이 풀려 있도록 취소가 끝나기를 잠깐 기다린다.
        # 연결이 끊겨 이 작업 자체가 취소된 중이어도 기다릴 수 있게 취소를 막는다.
        with anyio.CancelScope(shield=True):
            await asyncio.wait({producer}, timeout=STREAM_CANCEL_WAIT_SECONDS)


@router.post(
    "/sessions/{conversationRoomId}/messages/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_chat_message(
    conversation_room_id: ConversationRoomPath,
    body: ChatMessageStreamRequest,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> StreamingResponse:
    # 스트림을 열기 전에 막히는 요청은 메시지 전송과 같은 HTTP 오류로 응답한다
    try:
        confirmed = await use_cases.find_confirmed_turn(
            conversation_room_id,
            user_id=body.user_id,
            user_message_id=body.user_message_id,
        )
    except Exception as error:
        _raise_chat_error(error)
    if confirmed is not None:
        events = _single(confirmed)
    else:
        events = use_cases.stream_message(
            conversation_room_id,
            body.message,
            user_id=body.user_id,
            user_message_id=body.user_message_id,
        )
    return StreamingResponse(
        _sse_stream(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/sessions/{conversationRoomId}/analysis",
    response_model=AnalyzeChatSessionResponse,
)
async def analyze_chat_session(
    conversation_room_id: ConversationRoomPath,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> AnalyzeChatSessionResponse:
    try:
        analysis = await use_cases.analyze_session(conversation_room_id)
    except Exception as error:
        _raise_chat_error(error)
    return _analysis_response(analysis)


@router.patch(
    "/sessions/{conversationRoomId}/analysis",
    response_model=AnalyzeChatSessionResponse,
)
async def update_chat_analysis(
    conversation_room_id: ConversationRoomPath,
    body: UpdateChatAnalysisRequest,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> AnalyzeChatSessionResponse:
    try:
        analysis = await use_cases.update_analysis(
            conversation_room_id,
            user_id=body.user_id,
            summary=body.summary,
            taste_keywords=body.keywords.taste,
            interest_keywords=body.keywords.interest,
        )
    except Exception as error:
        _raise_chat_error(error)
    return _analysis_response(analysis)


@router.post(
    "/sessions/{conversationRoomId}/close",
    response_model=CloseChatSessionResponse,
)
async def close_chat_session(
    conversation_room_id: ConversationRoomPath,
    body: CloseChatSessionRequest,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
    recommendation_service: Annotated[
        RecommendationService,
        Depends(get_recommendation_service),
    ],
) -> CloseChatSessionResponse:
    try:
        closable = await use_cases.get_close_analysis(
            conversation_room_id,
            user_id=body.user_id,
        )
        recommendation_result = await recommendation_service.recommend_lists(
            {
                "userId": closable.state.user_id,
                "profile": recommendation_profile(closable),
            }
        )
        recommendations = RecommendationLists.model_validate(recommendation_result)
        finalized = await use_cases.close_session(conversation_room_id, user_id=body.user_id)
    except Exception as error:
        _raise_chat_error(error)
    return CloseChatSessionResponse(
        conversation_id=finalized.state.conversation_room_id,
        user_id=finalized.state.user_id,
        summary=finalized.summary,
        keywords=_scored_keywords(finalized),
        recommendations=recommendations,
    )
