from typing import Annotated

from fastapi import APIRouter, Depends, Path, status

from app.api.dependencies import get_chat_use_cases, get_recommendation_service
from app.api.schemas.chat import (
    AnalyzeChatSessionResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    CloseChatSessionRequest,
    CloseChatSessionResponse,
    CreateChatSessionRequest,
    CreateChatSessionResponse,
    UpdateChatAnalysisRequest,
)
from app.api.schemas.common import JAVA_LONG_MAX
from app.api.schemas.profile import ProfileItem, ProfileKeywords, TasteProfile
from app.api.schemas.recommendation import RecommendationLists
from app.application.chat_use_cases import (
    AnalysisAlreadyCorrectedError,
    AnalysisKeywordsCanOnlyBeDeletedError,
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
from app.application.conversation_state_store import SessionLifetimeExceededError
from app.application.ports.recommendation_service import RecommendationService
from app.core.errors import ApiError
from app.domain.conversation.models import SessionStatus
from app.domain.conversation.policy import MAX_TURNS, conversation_progress
from app.domain.profile.models import TasteField

router = APIRouter()

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


def _analysis_response(analysis: ProfileAnalysis) -> AnalyzeChatSessionResponse:
    return AnalyzeChatSessionResponse(
        profile={
            "userId": analysis.state.user_id,
            "summary": analysis.summary,
            "keywords": {
                "taste": list(analysis.taste_keywords),
                "interest": list(analysis.interest_keywords),
            },
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
    input_locked = completed.state.status == SessionStatus.INPUT_LOCKED
    return ChatMessageResponse(
        reply=completed.reply,
        created_at=completed.state.history[-1].created_at,
        expiration_at=completed.state.expiration_at,
        turn=completed.state.turn_count,
        max_turns=MAX_TURNS,
        progress=conversation_progress(completed.state),
        can_close=completed.readiness.sufficient,
        input_locked=input_locked,
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
        profile = _profile_response(closable)
        recommendation_result = await recommendation_service.recommend_lists(
            {
                "userId": closable.state.user_id,
                "profile": profile.model_dump(mode="json", by_alias=True),
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
        keywords=ProfileKeywords(
            taste=list(finalized.taste_keywords),
            interest=list(finalized.interest_keywords),
        ),
        recommendations=recommendations,
    )
