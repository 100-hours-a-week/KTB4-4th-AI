from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.schemas.chat import CreateChatSessionRequest
from app.api.schemas.profile import ProfileItem, TasteProfile
from app.api.schemas.recommendation import CreateRecommendationRequest


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
        "userId": "10293",
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
        {"userId": "10293", "existingProfile": profile_payload()}
    )

    assert request.user_id == "10293"
    assert request.model_dump(by_alias=True)["existingProfile"]["schemaVersion"] == "3.0"


def test_v1_chat_session_request_rejects_v2_onboarding_field() -> None:
    with pytest.raises(ValidationError):
        CreateChatSessionRequest.model_validate(
            {
                "userId": "10293",
                "existingProfile": None,
                "onboarding": {
                    "seedCategories": ["아웃도어"],
                    "excludeCategories": [],
                    "constraints": [],
                },
            }
        )


def test_taste_profile_requires_all_ten_arrays() -> None:
    payload = profile_payload()
    del payload["constraints"]

    with pytest.raises(ValidationError):
        TasteProfile.model_validate(payload)


def test_taste_profile_rejects_more_than_twelve_items() -> None:
    payload = profile_payload()
    payload["interests"] = [
        profile_item(str(index)).model_dump(by_alias=True) for index in range(13)
    ]

    with pytest.raises(ValidationError):
        TasteProfile.model_validate(payload)


def test_recommendation_request_applies_v1_defaults() -> None:
    request = CreateRecommendationRequest.model_validate(
        {"userId": "10293", "mode": "self", "profile": profile_payload()}
    )

    assert request.exclude_categories == []
    assert request.exclude_product_ids == []
    assert request.feedback_summary is None
    assert request.limit == 20
