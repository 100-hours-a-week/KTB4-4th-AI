from datetime import datetime

from pydantic import Field

from app.api.schemas.common import ApiModel, ConversationRoomId, MessageId, UserId
from app.api.schemas.profile import ProfileKeywords, ScoredProfileKeywords
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


class ChatMessageStreamRequest(ApiModel):
    user_id: UserId
    # BE가 저장한 사용자 메시지 ID. 같은 메시지를 다시 보낼 때 같은 값이다.
    user_message_id: MessageId
    message: str = Field(min_length=1, max_length=500)


class ChatStreamDelta(ApiModel):
    index: int = Field(ge=0)
    text: str


class ChatStreamError(ApiModel):
    code: str
    message: str
    retryable: bool


class ClientProfileAnalysis(ApiModel):
    user_id: UserId
    summary: str | None
    keywords: ScoredProfileKeywords
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
    keywords: ScoredProfileKeywords
    recommendations: RecommendationLists
