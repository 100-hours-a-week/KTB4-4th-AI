from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.schemas.chat import CreateChatSessionRequest, UpdateChatAnalysisRequest
from app.api.schemas.profile import ProfileItem, TasteProfile
from app.api.schemas.recommendation import RecommendedItem


def profile_item(value: str = "핸드드립") -> ProfileItem:
    timestamp = datetime(2026, 9, 16, tzinfo=UTC)
    return ProfileItem(
        value=value,
        confidence=0.9,
        linkRole="query",
        visibility="friends",
        intentType="want",
        deferralSignal=False,
        deferralReason=None,
        evidence="커피를 직접 내려 마셔요",
        taxonomyPath=["커피", "핸드드립"],
        firstSeenAt=timestamp,
        updatedAt=timestamp,
    )


def profile_payload() -> dict[str, object]:
    return {
        "schemaVersion": "3.0",
        "userId": 10293,
        "summary": None,
        "interests": [profile_item().model_dump(by_alias=True)],
        "hobbies": [],
        "preferences": [],
        "lifestyle": [],
        "wants": [],
        "unaffordable": [],
        "consumables": [],
        "owned": [],
        "dislikes": [],
        "constraints": [],
        "axes": [],
    }


def test_chat_session_request_accepts_backend_camel_case() -> None:
    request = CreateChatSessionRequest.model_validate(
        {"userId": 10293, "conversationRoomId": 45678}
    )

    assert request.user_id == 10293
    assert request.conversation_room_id == 45678


def test_v1_chat_session_request_rejects_onboarding_and_existing_profile() -> None:
    with pytest.raises(ValidationError):
        CreateChatSessionRequest.model_validate(
            {
                "userId": 10293,
                "conversationRoomId": 45678,
                "onboarding": {
                    "seedCategories": ["아웃도어"],
                    "excludeCategories": [],
                    "constraints": [],
                },
            }
        )

    with pytest.raises(ValidationError):
        CreateChatSessionRequest.model_validate(
            {
                "userId": 10293,
                "conversationRoomId": 45678,
                "existingProfile": profile_payload(),
            }
        )


def test_taste_profile_requires_all_ten_arrays() -> None:
    payload = profile_payload()
    del payload["constraints"]

    with pytest.raises(ValidationError):
        TasteProfile.model_validate(payload)


def test_user_id_rejects_quoted_number_to_keep_backend_contract_strict() -> None:
    with pytest.raises(ValidationError):
        CreateChatSessionRequest.model_validate({"userId": "10293", "conversationRoomId": 45678})

    with pytest.raises(ValidationError):
        CreateChatSessionRequest.model_validate({"userId": 10293, "conversationRoomId": "45678"})


def test_taste_profile_rejects_more_than_six_active_query_items() -> None:
    payload = profile_payload()
    payload["interests"] = [
        profile_item(str(index)).model_dump(by_alias=True) for index in range(7)
    ]

    with pytest.raises(ValidationError):
        TasteProfile.model_validate(payload)


def test_taste_profile_does_not_drop_safety_items_to_fit_working_set() -> None:
    payload = profile_payload()
    payload["constraints"] = [
        profile_item(str(index)).model_dump(by_alias=True) for index in range(13)
    ]

    profile = TasteProfile.model_validate(payload)

    assert len(profile.constraints) == 13


def test_analysis_update_requires_backend_user_id() -> None:
    request = UpdateChatAnalysisRequest.model_validate(
        {
            "userId": 10293,
            "summary": "캠핑을 즐기는 분입니다.",
            "keywords": {"taste": [], "interest": ["캠핑"]},
        }
    )

    assert request.user_id == 10293


def test_recommended_item_contains_backend_join_key_rank_score_and_reason() -> None:
    item = RecommendedItem.model_validate(
        {
            "platform": "coupang",
            "externalId": "12345",
            "score": 9.2,
            "reason": "캠핑 취향과 잘 맞는 상품이에요.",
        }
    )

    assert item.model_dump(by_alias=True) == {
        "platform": "coupang",
        "externalId": "12345",
        "score": 9.2,
        "reason": "캠핑 취향과 잘 맞는 상품이에요.",
    }


def test_analysis_update_rejects_missing_user_id() -> None:
    with pytest.raises(ValidationError):
        UpdateChatAnalysisRequest.model_validate(
            {
                "summary": "캠핑을 즐기는 분입니다.",
                "keywords": {"taste": [], "interest": ["캠핑"]},
            }
        )
