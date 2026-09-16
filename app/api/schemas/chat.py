from typing import Literal

from pydantic import Field

from app.api.schemas.common import ApiModel
from app.api.schemas.profile import ProfileKeywords, TasteProfile


class CreateChatSessionRequest(ApiModel):
    user_id: str = Field(min_length=1)
    existing_profile: TasteProfile | None


class CreateChatSessionResponse(ApiModel):
    session_id: str
    greeting: str
    max_turns: int = Field(ge=1)


class ChatMessageRequest(ApiModel):
    message: str = Field(min_length=1, max_length=500)


class ChatTokenEvent(ApiModel):
    text: str


class ChatDoneEvent(ApiModel):
    turn: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    can_close: bool
    item_count: int = Field(ge=0)
    input_locked: bool | None = None
    completion_reason: Literal["SUFFICIENT", "USER_EXIT", "MAX_CYCLES"] | None = None
    profile_completeness: Literal["sufficient", "partial"] | None = None
    last_turn_extraction_failed: bool | None = None


class CloseChatSessionResponse(ApiModel):
    profile: TasteProfile
    summary: str | None
    keywords: ProfileKeywords
    correction_available: bool
    completion_reason: Literal["SUFFICIENT", "USER_EXIT", "MAX_CYCLES"]
    profile_completeness: Literal["sufficient", "partial"]
    missing_signals: list[str] = Field(default_factory=list)
