from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_conversation_state_store, get_recommendation_service
from app.api.schemas.recommendation import (
    CreateRecommendationListsRequest,
    CreateRecommendationRequest,
    RecommendationLists,
    RecommendationResult,
)
from app.application.conversation_state_store import ConversationStateStore
from app.application.ports.recommendation_service import RecommendationService

router = APIRouter()


@router.post(
    "/jobs",
    response_model=RecommendationLists,
)
async def create_recommendation_lists(
    body: CreateRecommendationListsRequest,
    service: Annotated[RecommendationService, Depends(get_recommendation_service)],
    states: Annotated[ConversationStateStore, Depends(get_conversation_state_store)],
) -> RecommendationLists:
    result = await service.recommend_lists(body.model_dump(mode="json", by_alias=True))
    response = RecommendationLists.model_validate(result)
    if body.session_id is not None:
        await states.delete(body.session_id)
    return response


@router.post("", response_model=RecommendationResult)
async def create_recommendation(
    body: CreateRecommendationRequest,
    service: Annotated[RecommendationService, Depends(get_recommendation_service)],
) -> RecommendationResult:
    result = await service.recommend(body.model_dump(mode="json", by_alias=True))
    return RecommendationResult.model_validate(result)
