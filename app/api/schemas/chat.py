from datetime import datetime
from typing import Literal

from pydantic import Field

from app.api.schemas.common import ApiModel, ConversationRoomId, UserId
from app.api.schemas.profile import ProfileKeywords, TasteProfile


class CreateChatSessionRequest(ApiModel):
    user_id: UserId
    conversation_room_id: ConversationRoomId


class CreateChatSessionResponse(ApiModel):
    session_id: ConversationRoomId
    greeting: str
    max_turns: int = Field(ge=1)


class ChatHistoryMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str


class GetChatSessionResponse(ApiModel):
    session_id: ConversationRoomId
    user_id: UserId
    status: Literal["active", "input_locked", "review"]
    turn: int = Field(ge=0)
    max_turns: int = Field(ge=1)
    messages: list[ChatHistoryMessage]
    item_count: int = Field(ge=0)
    input_locked: bool
    analysis_available: bool
    can_close: bool
    completion_reason: Literal["SUFFICIENT", "USER_EXIT", "MAX_CYCLES"] | None = None
    profile_completeness: Literal["sufficient", "partial"] | None = None
    last_active_at: datetime


class ChatMessageRequest(ApiModel):
    message: str = Field(min_length=1, max_length=500)


class ChatMessageResponse(ApiModel):
    reply: str
    turn: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    can_close: bool
    item_count: int = Field(ge=0)
    input_locked: bool
    completion_reason: Literal["SUFFICIENT", "USER_EXIT", "MAX_CYCLES"] | None = None
    profile_completeness: Literal["sufficient", "partial"] | None = None
    last_turn_extraction_failed: bool


class ProfileAnalysisResponse(ApiModel):
    profile: TasteProfile
    summary: str | None
    keywords: ProfileKeywords
    profile_completeness: Literal["sufficient", "partial"]
    missing_signals: list[str] = Field(default_factory=list)


class AnalyzeChatSessionResponse(ProfileAnalysisResponse):
    correction_available: Literal[True] = True


class CloseChatSessionResponse(ProfileAnalysisResponse):
    completion_reason: Literal["SUFFICIENT", "USER_EXIT", "MAX_CYCLES"]
