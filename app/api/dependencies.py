from typing import cast

from fastapi import Request

from app.application.chat_use_cases import ChatUseCases
from app.application.conversation_state_store import ConversationStateStore
from app.application.ports.recommendation_service import RecommendationService
from app.core.errors import ApiError


def get_chat_use_cases(request: Request) -> ChatUseCases:
    return cast(ChatUseCases, request.app.state.chat_use_cases)


def get_conversation_state_store(request: Request) -> ConversationStateStore:
    return cast(ConversationStateStore, request.app.state.conversation_state_store)


def get_recommendation_service(request: Request) -> RecommendationService:
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise ApiError(
            status_code=503,
            code="RECOMMENDATION_UNAVAILABLE",
            message="추천 파이프라인이 아직 준비되지 않았습니다.",
            retryable=False,
        )
    return cast(RecommendationService, service)
