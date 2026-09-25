from datetime import datetime

from pydantic import Field

from app.api.schemas.common import ApiModel, ConversationRoomId, UserId
from app.api.schemas.profile import ProfileKeywords
from app.api.schemas.recommendation import RecommendationLists


class CreateChatSessionRequest(ApiModel):
    user_id: UserId
    conversation_room_id: ConversationRoomId


class CreateChatSessionResponse(ApiModel):
    conversation_room_id: ConversationRoomId
    greeting: str
    created_at: datetime
    expiration_at: datetime
    max_turns: int = Field(ge=1)


class ChatMessageRequest(ApiModel):
    user_id: UserId
    message: str = Field(min_length=1, max_length=500)


class ChatMessageResponse(ApiModel):
    reply: str
    created_at: datetime
    expiration_at: datetime
    turn: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    progress: int = Field(ge=0, le=100)
    can_close: bool
    input_locked: bool


class ClientProfileAnalysis(ApiModel):
    user_id: UserId
    summary: str | None
    keywords: ProfileKeywords
    correction_available: bool


class AnalyzeChatSessionResponse(ApiModel):
    profile: ClientProfileAnalysis


class UpdateChatAnalysisRequest(ApiModel):
    user_id: UserId
    summary: str | None
    keywords: ProfileKeywords


class CloseChatSessionRequest(ApiModel):
    user_id: UserId


class CloseChatSessionResponse(ApiModel):
    conversation_id: ConversationRoomId
    user_id: UserId
    summary: str | None
    keywords: ProfileKeywords
    recommendations: RecommendationLists
