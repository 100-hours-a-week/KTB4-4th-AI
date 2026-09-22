from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status

from app.api.dependencies import get_chat_use_cases
from app.api.schemas.chat import (
    AnalyzeChatSessionResponse,
    ChatHistoryMessage,
    ChatMessageRequest,
    ChatMessageResponse,
    CloseChatSessionResponse,
    CreateChatSessionRequest,
    CreateChatSessionResponse,
    GetChatSessionResponse,
)
from app.api.schemas.common import JAVA_LONG_MAX
from app.api.schemas.profile import ProfileItem, ProfileKeywords, TasteProfile
from app.application.chat_use_cases import (
    AnalysisOutdatedError,
    AnalysisRequiredError,
    ChatUseCases,
    ProfileAnalysis,
    ProfileTooSparseError,
    ResponseModelUnavailableError,
    SessionClosedError,
    SessionExistsError,
    SessionInputLockedError,
    SessionNotFoundError,
    TurnInProgressError,
)
from app.core.errors import ApiError
from app.domain.conversation.models import SessionStatus
from app.domain.conversation.policy import MAX_TURNS, recommendation_readiness
from app.domain.profile.models import TasteField

router = APIRouter()

ConversationRoomPath = Annotated[
    int,
    Path(alias="conversationRoomId", ge=1, le=JAVA_LONG_MAX),
]
UserIdQuery = Annotated[
    int,
    Query(alias="userId", ge=1, le=JAVA_LONG_MAX),
]


def _raise_chat_error(error: Exception) -> None:
    if isinstance(error, SessionNotFoundError):
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
            message="입력이 종료되었습니다. 현재 취향 분석 결과를 먼저 확인해 주세요.",
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
    if isinstance(error, ResponseModelUnavailableError):
        raise ApiError(
            status_code=502,
            code="LLM_UNAVAILABLE",
            message="대화 모델을 호출할 수 없습니다.",
            retryable=True,
        ) from error
    raise error


def _profile_response(result: ProfileAnalysis) -> TasteProfile:
    grouped: dict[TasteField, list[ProfileItem]] = {field: [] for field in TasteField}
    for signal in result.profile.active_signals():
        grouped[signal.field].append(
            ProfileItem(
                value=signal.value,
                confidence=signal.confidence,
                link_role=signal.link_role.value,
                visibility=signal.visibility.value,
                intent_type=signal.intent_type.value if signal.intent_type else None,
                deferral_signal=signal.deferral_signal,
                deferral_reason=(signal.deferral_reason.value if signal.deferral_reason else None),
                evidence=signal.evidence,
                taxonomy_path=None,
                first_seen_at=signal.first_seen_at,
                updated_at=signal.updated_at,
            )
        )
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
        session_id=started.state.session_id,
        greeting=started.greeting,
        max_turns=MAX_TURNS,
    )


@router.get(
    "/sessions/{conversationRoomId}",
    response_model=GetChatSessionResponse,
)
async def get_chat_session(
    conversation_room_id: ConversationRoomPath,
    user_id: UserIdQuery,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> GetChatSessionResponse:
    try:
        state = await use_cases.get_session(conversation_room_id, user_id=user_id)
    except Exception as error:
        _raise_chat_error(error)
    readiness = recommendation_readiness(state)
    input_locked = state.status == SessionStatus.INPUT_LOCKED
    analysis_available = (
        state.analysis_turn_count is not None and state.analysis_turn_count == state.turn_count
    )
    review_or_locked = state.status in {SessionStatus.INPUT_LOCKED, SessionStatus.REVIEW}
    return GetChatSessionResponse(
        session_id=state.session_id,
        user_id=state.user_id,
        status=state.status.value,
        turn=state.turn_count,
        max_turns=MAX_TURNS,
        messages=[
            ChatHistoryMessage(role=message.role, content=message.content)
            for message in state.history
        ],
        item_count=len(state.profile.active_signals()),
        input_locked=input_locked,
        analysis_available=analysis_available,
        can_close=readiness.sufficient,
        completion_reason=(state.completion_reason.value if state.completion_reason else None),
        profile_completeness=("sufficient" if readiness.sufficient else "partial")
        if review_or_locked
        else None,
        last_active_at=state.last_active_at,
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
        completed = await use_cases.send_message(conversation_room_id, body.message)
    except Exception as error:
        _raise_chat_error(error)
    input_locked = completed.state.status == SessionStatus.INPUT_LOCKED
    return ChatMessageResponse(
        reply=completed.reply,
        turn=completed.state.turn_count,
        max_turns=MAX_TURNS,
        can_close=completed.readiness.sufficient,
        item_count=len(completed.state.profile.active_signals()),
        input_locked=input_locked,
        completion_reason=(
            completed.state.completion_reason.value
            if completed.state.completion_reason is not None
            else None
        ),
        profile_completeness=("sufficient" if completed.readiness.sufficient else "partial")
        if input_locked
        else None,
        last_turn_extraction_failed=completed.extraction_failed,
    )


@router.delete(
    "/sessions/{conversationRoomId}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_session(
    conversation_room_id: ConversationRoomPath,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> Response:
    try:
        await use_cases.delete_session(conversation_room_id)
    except Exception as error:
        _raise_chat_error(error)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
    return AnalyzeChatSessionResponse(
        profile=_profile_response(analysis),
        summary=analysis.summary,
        keywords=ProfileKeywords(
            taste=list(analysis.taste_keywords),
            interest=list(analysis.interest_keywords),
        ),
        profile_completeness=("sufficient" if analysis.readiness.sufficient else "partial"),
        missing_signals=list(analysis.readiness.missing_signals),
    )


@router.post(
    "/sessions/{conversationRoomId}/close",
    response_model=CloseChatSessionResponse,
)
async def close_chat_session(
    conversation_room_id: ConversationRoomPath,
    use_cases: Annotated[ChatUseCases, Depends(get_chat_use_cases)],
) -> CloseChatSessionResponse:
    try:
        finalized = await use_cases.close_session(conversation_room_id)
    except Exception as error:
        _raise_chat_error(error)
    return CloseChatSessionResponse(
        profile=_profile_response(finalized),
        summary=finalized.summary,
        keywords=ProfileKeywords(
            taste=list(finalized.taste_keywords),
            interest=list(finalized.interest_keywords),
        ),
        completion_reason=finalized.state.completion_reason.value,
        profile_completeness=("sufficient" if finalized.readiness.sufficient else "partial"),
        missing_signals=list(finalized.readiness.missing_signals),
    )
