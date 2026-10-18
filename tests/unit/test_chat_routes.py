from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any

from fastapi.testclient import TestClient

from app.application.ports.model_gateway import Message
from app.core.config import Settings
from app.domain.conversation.models import SessionStatus
from app.domain.conversation.policy import MAX_TURNS
from app.domain.profile.models import (
    EvidenceType,
    IntentType,
    LinkRole,
    ProfileSignal,
    SignalStatus,
    TasteField,
    Visibility,
)
from app.infrastructure.persistence import InMemorySessionStore
from app.main import create_app


class FakeModelGateway:
    def __init__(
        self,
        *,
        completions: list[str] | None = None,
        structured_results: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.completions = completions or [
            "안녕하세요! 요즘 어떻게 지내세요?",
            "캠핑 좋죠. 주로 어디로 다니세요?",
            "캠핑을 즐기고 직접 장비를 고르는 분입니다.",
        ]
        self.structured_results = structured_results or []

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        return self.completions.pop(0)

    def stream(self, messages: Sequence[Message], *, model: str) -> AsyncIterator[str]:
        raise NotImplementedError

    async def structured(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.structured_results:
            return self.structured_results.pop(0)
        return {
            "items": [
                {
                    "field": "hobbies",
                    "value": "캠핑",
                    "confidence": 0.9,
                    "evidence": "주말마다 캠핑 가요",
                    "evidenceType": "explicit",
                    "intentType": "both",
                    "deferralReason": None,
                }
            ],
            "axes": [],
            "drop": [],
        }


class FakeRecommendationService:
    def __init__(self) -> None:
        self.lists_payload: Mapping[str, Any] | None = None

    async def recommend_lists(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.lists_payload = payload
        return {
            "generatedAt": "2026-09-21T07:00:00Z",
            "self": {"items": _recommended_items()},
            "gift": {"items": _recommended_items()},
        }


def _recommended_items() -> list[dict[str, object]]:
    return [
        {
            "platform": "coupang",
            "externalId": "12345",
            "score": 0.92,
            "reason": "캠핑 취향과 잘 맞는 상품이에요.",
        }
    ]


def test_chat_http_lifecycle_matches_v1_contract() -> None:
    recommendation_service = FakeRecommendationService()
    app = create_app(
        Settings(
            app_env="test",
            service_token="test-token",
            redis_url=None,
            response_model_base_url=None,
            extraction_model_base_url=None,
        ),
        model_gateway=FakeModelGateway(),
        session_store=InMemorySessionStore(),
        recommendation_service=recommendation_service,
    )
    headers = {"Authorization": "Bearer test-token", "X-Request-ID": "req_route_test"}

    with TestClient(app) as client:
        created = client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )
        message = client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "주말마다 캠핑 가요"},
        )
        analysis = client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        closed = client.post(
            "/v1/chat/sessions/45678/close", headers=headers, json={"userId": 10293}
        )
        closed_again = client.post(
            "/v1/chat/sessions/45678/close", headers=headers, json={"userId": 10293}
        )
    assert created.status_code == 201
    created_body = created.json()
    created_at = datetime.fromisoformat(created_body.pop("createdAt"))
    expiration_at = datetime.fromisoformat(created_body.pop("expirationAt"))
    assert expiration_at > created_at
    assert created_body == {
        "conversationRoomId": 45678,
        "greeting": "안녕하세요! 요즘 어떻게 지내세요?",
        "maxTurns": 20,
    }
    assert message.status_code == 200
    message_body = message.json()
    message_created_at = datetime.fromisoformat(message_body.pop("createdAt"))
    message_expiration_at = datetime.fromisoformat(message_body.pop("expirationAt"))
    assert message_expiration_at > message_created_at
    assert message_body == {
        "reply": "캠핑 좋죠. 주로 어디로 다니세요?",
        "turn": 1,
        "maxTurns": 20,
        "progress": 21,
        "canClose": False,
        "inputLocked": False,
    }
    assert analysis.status_code == 200
    assert analysis.json()["profile"] == {
        "userId": 10293,
        "summary": "캠핑을 즐기고 직접 장비를 고르는 분입니다.",
        "keywords": {"taste": [], "interest": [{"value": "캠핑", "score": 0.95}]},
        "correctionAvailable": True,
    }
    assert closed.status_code == 200
    assert set(closed.json()) == {
        "conversationId",
        "userId",
        "summary",
        "keywords",
        "recommendations",
    }
    assert closed.json()["conversationId"] == 45678
    assert closed.json()["summary"] == analysis.json()["profile"]["summary"]
    assert closed.json()["keywords"] == analysis.json()["profile"]["keywords"]
    assert closed.json()["recommendations"]["generatedAt"] == "2026-09-21T07:00:00Z"
    assert set(closed.json()["recommendations"]) == {"generatedAt", "self", "gift"}
    assert closed_again.status_code == 409
    assert closed_again.json()["code"] == "SESSION_CLOSED"
    assert created.headers["X-Request-ID"] == "req_route_test"


def test_v1_routes_require_service_token() -> None:
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=FakeModelGateway(),
        session_store=InMemorySessionStore(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/sessions",
            json={"userId": 10293, "conversationRoomId": 45678},
        )

    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_removed_session_restore_and_delete_routes_are_not_exposed() -> None:
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=FakeModelGateway(),
        session_store=InMemorySessionStore(),
    )

    paths = app.openapi()["paths"]

    assert "/v1/chat/sessions/{conversationRoomId}" not in paths


def test_analysis_patch_directly_updates_summary_and_keywords() -> None:
    gateway = FakeModelGateway(
        completions=[
            "안녕하세요! 요즘 어떻게 지내세요?",
            "캠핑을 좋아하시는군요.",
            "캠핑을 즐기는 분입니다.",
        ],
        structured_results=[
            {
                "items": [
                    {
                        "field": "hobbies",
                        "value": "캠핑",
                        "confidence": 0.9,
                        "evidence": "캠핑을 좋아해요",
                        "evidenceType": "explicit",
                        "intentType": "both",
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [],
            },
        ],
    )
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=gateway,
        session_store=InMemorySessionStore(),
        recommendation_service=FakeRecommendationService(),
    )
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )
        client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "캠핑을 좋아해요"},
        )
        patch_without_analysis = client.patch(
            "/v1/chat/sessions/45678/analysis",
            headers=headers,
            json={
                "userId": 10293,
                "summary": "캠핑을 즐기는 사람입니다.",
                "keywords": {"taste": ["가벼운 장비"], "interest": ["캠핑"]},
            },
        )
        client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        keyword_addition = client.patch(
            "/v1/chat/sessions/45678/analysis",
            headers=headers,
            json={
                "userId": 10293,
                "summary": "캠핑과 자연을 좋아합니다.",
                "keywords": {"taste": [], "interest": ["캠핑", "자연"]},
            },
        )
        message_in_review = client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "요약을 수정할게요"},
        )
        updated = client.patch(
            "/v1/chat/sessions/45678/analysis",
            headers=headers,
            json={
                "userId": 10293,
                "summary": "주말마다 자연에서 쉬는 것을 좋아합니다.",
                "keywords": {"taste": [], "interest": []},
            },
        )
        second_patch = client.patch(
            "/v1/chat/sessions/45678/analysis",
            headers=headers,
            json={
                "userId": 10293,
                "summary": "다시 수정합니다.",
                "keywords": {"taste": [], "interest": []},
            },
        )
        closed = client.post(
            "/v1/chat/sessions/45678/close", headers=headers, json={"userId": 10293}
        )

    assert patch_without_analysis.status_code == 409
    assert patch_without_analysis.json()["code"] == "ANALYSIS_REQUIRED"
    assert keyword_addition.status_code == 422
    assert keyword_addition.json()["code"] == "KEYWORDS_DELETE_ONLY"
    assert message_in_review.status_code == 409
    assert message_in_review.json()["code"] == "INPUT_LOCKED"
    assert updated.status_code == 200
    assert updated.json()["profile"]["summary"] == "주말마다 자연에서 쉬는 것을 좋아합니다."
    assert updated.json()["profile"]["keywords"] == {
        "taste": [],
        "interest": [],
    }
    assert updated.json()["profile"]["correctionAvailable"] is False
    assert second_patch.status_code == 409
    assert second_patch.json()["code"] == "ANALYSIS_ALREADY_CORRECTED"
    assert closed.status_code == 200
    assert closed.json()["summary"] == updated.json()["profile"]["summary"]
    assert closed.json()["keywords"] == updated.json()["profile"]["keywords"]
    recommendation_payload = app.state.recommendation_service.lists_payload
    assert recommendation_payload is not None
    assert recommendation_payload["profile"]["hobbies"][0]["value"] == "캠핑"


def test_routes_include_recommendation_contract_but_exclude_v2_and_unagreed_apis() -> None:
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=FakeModelGateway(),
        session_store=InMemorySessionStore(),
    )

    paths = app.openapi()["paths"]

    assert "/v1/recommendations/jobs" not in paths
    assert "/v1/recommendations/jobs/{jobId}" not in paths
    assert "/v1/recommendations" not in paths
    assert "/v1/chat/sessions/{conversationRoomId}/analysis" in paths
    assert "patch" in paths["/v1/chat/sessions/{conversationRoomId}/analysis"]
    assert "/v1/chat/sessions/{conversationRoomId}" not in paths
    assert "/v1/recommendations/{recommendationId}/feedback" not in paths
    assert "/v1/catalog/coverage" not in paths


def test_close_reports_unavailable_until_pipeline_is_injected() -> None:
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=FakeModelGateway(),
        session_store=InMemorySessionStore(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/sessions/45678/close",
            headers={"Authorization": "Bearer test-token"},
            json={"userId": 10293},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "RECOMMENDATION_UNAVAILABLE"


def test_close_returns_initial_recommendations_without_second_request() -> None:
    recommendation_service = FakeRecommendationService()
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=FakeModelGateway(),
        session_store=InMemorySessionStore(),
        recommendation_service=recommendation_service,
    )
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )
        client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "주말마다 캠핑 가요"},
        )
        client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        response = client.post(
            "/v1/chat/sessions/45678/close", headers=headers, json={"userId": 10293}
        )
    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    assert recommendations["generatedAt"] == "2026-09-21T07:00:00Z"
    assert recommendations["self"]["items"] == [
        {
            "platform": "coupang",
            "externalId": "12345",
            "score": 0.92,
            "reason": "캠핑 취향과 잘 맞는 상품이에요.",
        }
    ]
    assert "jobId" not in recommendations
    assert recommendation_service.lists_payload is not None
    assert recommendation_service.lists_payload["userId"] == 10293
    assert "sessionId" not in recommendation_service.lists_payload
    # 온보딩 값 없이 만든 대화방은 온보딩 조건 없이 추천한다.
    assert "onboarding" not in recommendation_service.lists_payload


ONBOARDING_USER_PROFILE = {
    "nickname": "니듀",
    "birthDate": "2000-01-01",
    "gender": "FEMALE",
    "interestCategoryCodes": ["COFFEE_TEA", "TRAVEL"],
    "allergyCodes": ["PEANUT"],
    "giftExclusionCodes": ["ALCOHOL"],
}


def _onboarding_app(
    recommendation_service: FakeRecommendationService,
    gateway: FakeModelGateway | None = None,
):
    return create_app(
        Settings(
            app_env="test",
            service_token="test-token",
            redis_url=None,
            response_model_base_url=None,
            extraction_model_base_url=None,
        ),
        model_gateway=gateway or FakeModelGateway(),
        session_store=InMemorySessionStore(),
        recommendation_service=recommendation_service,
    )


def test_close_passes_onboarding_from_session_creation_to_recommendation() -> None:
    recommendation_service = FakeRecommendationService()
    app = _onboarding_app(recommendation_service)
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        created = client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={
                "userId": 1,
                "conversationRoomId": 42,
                "userProfile": ONBOARDING_USER_PROFILE,
            },
        )
        client.post(
            "/v1/chat/sessions/42/messages",
            headers=headers,
            json={"userId": 1, "message": "주말마다 캠핑 가요"},
        )
        client.post("/v1/chat/sessions/42/analysis", headers=headers)
        closed = client.post("/v1/chat/sessions/42/close", headers=headers, json={"userId": 1})

    assert created.status_code == 201
    assert closed.status_code == 200
    assert recommendation_service.lists_payload is not None
    assert recommendation_service.lists_payload["onboarding"] == {
        "displayName": "니듀",
        "interestCategoryCodes": ["COFFEE_TEA", "TRAVEL"],
        "allergyCodes": ["PEANUT"],
        "giftExclusionCodes": ["ALCOHOL"],
    }


def test_create_session_rejects_unknown_onboarding_code() -> None:
    """모르는 알레르기 코드를 버리고 진행하면 그 알레르기가 추천에서 빠진다."""
    app = _onboarding_app(FakeRecommendationService())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/sessions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "userId": 1,
                "conversationRoomId": 42,
                "userProfile": {**ONBOARDING_USER_PROFILE, "allergyCodes": ["SESAME"]},
            },
        )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def _extracted(field: str, value: str, confidence: float) -> dict[str, object]:
    return {
        "field": field,
        "value": value,
        "confidence": confidence,
        "evidence": value,
        "evidenceType": "explicit",
        "intentType": "both",
        "deferralReason": None,
    }


def test_analysis_keywords_are_ordered_by_score_and_patch_keeps_that_order() -> None:
    gateway = FakeModelGateway(
        completions=[
            "안녕하세요! 요즘 어떻게 지내세요?",
            "좋은 취미가 많으시네요.",
            "산책과 사진을 즐기는 분입니다.",
        ],
        structured_results=[
            {
                "items": [
                    _extracted("interests", "음악", 0.6),
                    _extracted("interests", "사진", 0.8),
                    _extracted("hobbies", "산책", 0.95),
                ],
                "axes": [],
                "drop": [],
            },
        ],
    )
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=gateway,
        session_store=InMemorySessionStore(),
        recommendation_service=FakeRecommendationService(),
    )
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )
        client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "음악, 사진, 산책 다 좋아해요"},
        )
        analysis = client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        updated = client.patch(
            "/v1/chat/sessions/45678/analysis",
            headers=headers,
            json={
                "userId": 10293,
                "summary": "산책과 음악을 즐기는 분입니다.",
                "keywords": {"taste": [], "interest": ["음악", "산책"]},
            },
        )

    assert analysis.status_code == 200
    assert analysis.json()["profile"]["keywords"]["interest"] == [
        {"value": "산책", "score": 1.0},
        {"value": "사진", "score": 0.8},
        {"value": "음악", "score": 0.6},
    ]
    assert updated.status_code == 200
    assert updated.json()["profile"]["keywords"]["interest"] == [
        {"value": "산책", "score": 1.0},
        {"value": "음악", "score": 0.6},
    ]


def test_sparse_profile_analysis_expires_session_so_it_can_restart() -> None:
    gateway = FakeModelGateway(
        completions=[
            "안녕하세요! 요즘 어떻게 지내세요?",
            "그렇군요. 다른 이야기도 해볼까요?",
            "다시 만나서 반가워요!",
        ],
        structured_results=[{"items": [], "axes": [], "drop": []}],
    )
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=gateway,
        session_store=InMemorySessionStore(),
        recommendation_service=FakeRecommendationService(),
    )
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )
        client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "몰라요"},
        )
        analysis = client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        message_after_expiry = client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "안녕"},
        )
        restarted = client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )

    assert analysis.status_code == 422
    assert analysis.json()["code"] == "PROFILE_TOO_SPARSE"
    assert message_after_expiry.status_code == 404
    assert message_after_expiry.json()["code"] == "SESSION_NOT_FOUND"
    assert restarted.status_code == 201
    assert restarted.json()["greeting"] == "다시 만나서 반가워요!"


def _max_turn_profile_signal(field: TasteField, value: str) -> ProfileSignal:
    now = datetime(2026, 9, 18)
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=0.9,
        link_role=LinkRole.QUERY,
        visibility=Visibility.FRIENDS,
        intent_type=IntentType.BOTH,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=now,
        updated_at=now,
        status=SignalStatus.ACTIVE,
    )


def _analyze_after_max_turns(signals: list[ProfileSignal]) -> tuple[Any, Any]:
    gateway = FakeModelGateway(
        completions=["안녕하세요! 요즘 어떻게 지내세요?", "캠핑을 좋아하는 친구예요."],
    )
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=gateway,
        session_store=InMemorySessionStore(),
        recommendation_service=FakeRecommendationService(),
    )
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )
        store = app.state.conversation_state_store

        async def reach_max_turns() -> None:
            state = await store.load(45678)
            state.turn_count = MAX_TURNS
            state.status = SessionStatus.INPUT_LOCKED
            state.profile.signals = signals
            await store.save(state)

        client.portal.call(reach_max_turns)
        analysis = client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        message_after_analysis = client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"userId": 10293, "message": "안녕"},
        )
    return analysis, message_after_analysis


def test_max_turn_analysis_without_taste_or_interest_is_too_sparse() -> None:
    analysis, message_after_analysis = _analyze_after_max_turns(
        [_max_turn_profile_signal(TasteField.DISLIKES, "향수")]
    )

    assert analysis.status_code == 422
    assert analysis.json()["code"] == "PROFILE_TOO_SPARSE"
    assert message_after_analysis.status_code == 404


def test_max_turn_analysis_with_interest_returns_profile() -> None:
    analysis, _ = _analyze_after_max_turns([_max_turn_profile_signal(TasteField.INTERESTS, "캠핑")])

    assert analysis.status_code == 200
    assert analysis.json()["profile"]["keywords"]["interest"][0]["value"] == "캠핑"


def test_recommendation_profile_carries_every_stored_item_beyond_exposure_limits() -> None:
    from app.api.routes.v1_chat import _profile_response, recommendation_profile
    from app.application.chat_use_cases import ProfileAnalysis
    from app.domain.conversation.models import ConversationState
    from app.domain.conversation.policy import recommendation_readiness
    from app.domain.profile.merger import ProfileMerger
    from app.domain.profile.models import (
        EvidenceType,
        ExtractedItem,
        ExtractionDelta,
        ProfileState,
        TasteField,
    )

    tastes = ["무채색 옷", "혼자 가는 여행", "조용한 카페", "고소한 커피", "미니멀한 디자인"]
    utterance = ", ".join(tastes)
    profile = (
        ProfileMerger()
        .merge(
            ProfileState(),
            ExtractionDelta(
                items=tuple(
                    ExtractedItem(
                        field=TasteField.PREFERENCES,
                        value=value,
                        confidence=0.9,
                        evidence=value,
                        evidence_type=EvidenceType.EXPLICIT,
                    )
                    for value in tastes
                )
            ),
            utterance=utterance,
            now=datetime(2026, 9, 30),
            source_turn=1,
        )
        .profile
    )
    state = ConversationState(user_id=1, conversation_room_id=1, profile=profile)
    analysis = ProfileAnalysis(
        state=state,
        profile=profile,
        summary=None,
        taste_keywords=(),
        interest_keywords=(),
        keyword_scores={},
        readiness=recommendation_readiness(state),
    )

    assert len(_profile_response(analysis).preferences) == 3
    assert [item["value"] for item in recommendation_profile(analysis)["preferences"]] == tastes


class RecordingModelGateway(FakeModelGateway):
    """응답 모델에 넘어간 메시지를 호출 순서대로 남긴다."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.completed_messages: list[Sequence[Message]] = []

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        self.completed_messages.append(messages)
        return await super().complete(messages, model=model)


def test_first_greeting_and_replies_call_the_user_by_name_from_user_profile() -> None:
    gateway = RecordingModelGateway()
    app = _onboarding_app(FakeRecommendationService(), gateway)
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        created = client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={
                "userId": 1,
                "conversationRoomId": 42,
                "userProfile": {**ONBOARDING_USER_PROFILE, "nickname": "서동휘"},
            },
        )
        client.post(
            "/v1/chat/sessions/42/messages",
            headers=headers,
            json={"userId": 1, "message": "주말마다 캠핑 가요"},
        )

    assert created.status_code == 201
    opening, reply = gateway.completed_messages[:2]
    # 대화방을 만드는 첫인사부터 성을 뗀 이름으로 부르게 한다. 원문은 대화에 넣지 않는다.
    for messages in (opening, reply):
        contents = "\n".join(message["content"] for message in messages)
        assert '"동휘님"이라고 부른다' in contents
        assert "서동휘" not in contents


def test_first_greeting_does_not_name_the_user_without_a_usable_name() -> None:
    gateway = RecordingModelGateway()
    app = _onboarding_app(FakeRecommendationService(), gateway)

    with TestClient(app) as client:
        client.post(
            "/v1/chat/sessions",
            headers={"Authorization": "Bearer test-token"},
            json={
                "userId": 1,
                "conversationRoomId": 42,
                "userProfile": {**ONBOARDING_USER_PROFILE, "nickname": "."},
            },
        )

    contents = "\n".join(message["content"] for message in gateway.completed_messages[0])
    assert '님"이라고 부른다' not in contents


def test_summary_intro_is_corrected_to_the_full_username() -> None:
    gateway = RecordingModelGateway(
        completions=[
            "안녕하세요! 요즘 어떻게 지내세요?",
            "캠핑 좋죠. 주로 어디로 다니세요?",
            # 모델이 원문 대신 대화 호칭으로 썼다.
            "니쥬가 보기에 동휘님은 주말마다 캠핑을 떠나는 사람이에요.",
        ]
    )
    app = _onboarding_app(FakeRecommendationService(), gateway)
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={
                "userId": 1,
                "conversationRoomId": 42,
                "userProfile": {**ONBOARDING_USER_PROFILE, "nickname": "서동휘"},
            },
        )
        client.post(
            "/v1/chat/sessions/42/messages",
            headers=headers,
            json={"userId": 1, "message": "주말마다 캠핑 가요"},
        )
        analysis = client.post("/v1/chat/sessions/42/analysis", headers=headers)

    assert analysis.status_code == 200
    assert analysis.json()["profile"]["summary"] == (
        "니쥬가 보기에 서동휘님은 주말마다 캠핑을 떠나는 사람이에요."
    )
    summary_prompt = gateway.completed_messages[2][0]["content"]
    assert '첫 문장은 "니쥬가 보기에 서동휘님은"으로 시작해서' in summary_prompt
