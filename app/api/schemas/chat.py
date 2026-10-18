from datetime import date, datetime

from pydantic import Field

from app.api.schemas.common import ApiModel, ConversationRoomId, MessageId, UserId
from app.api.schemas.profile import ProfileKeywords, ScoredProfileKeywords
from app.api.schemas.recommendation import RecommendationLists
from app.domain.profile.onboarding import (
    Allergy,
    Gender,
    GiftExclusion,
    InterestCategory,
    OnboardingProfile,
)


class OnboardingUserProfile(ApiModel):
    """온보딩에서 받은 값. 코드 배열은 값이 없어도 []로 보낸다. 모르는 코드는 400이다."""

    nickname: str | None = Field(default=None, max_length=50)
    birth_date: date | None = None
    gender: Gender | None = None
    interest_category_codes: list[InterestCategory] = Field(max_length=len(InterestCategory))
    allergy_codes: list[Allergy] = Field(max_length=len(Allergy))
    gift_exclusion_codes: list[GiftExclusion] = Field(max_length=len(GiftExclusion))

    def to_domain(self) -> OnboardingProfile:
        return OnboardingProfile(
            nickname=self.nickname,
            birth_date=self.birth_date,
            gender=self.gender,
            interests=tuple(self.interest_category_codes),
            allergies=tuple(self.allergy_codes),
            gift_exclusions=tuple(self.gift_exclusion_codes),
        )


class CreateChatSessionRequest(ApiModel):
    user_id: UserId
    conversation_room_id: ConversationRoomId
    # 없으면 온보딩 조건 없이 추천한다.
    user_profile: OnboardingUserProfile | None = None


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
