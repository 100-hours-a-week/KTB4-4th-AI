from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any

from fastapi.testclient import TestClient

from app.application.ports.model_gateway import Message
from app.core.config import Settings
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
            "goalAssessment": {"goal": "INTEREST", "status": "found"},
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
        "progress": 25,
        "canClose": False,
        "inputLocked": False,
    }
    assert analysis.status_code == 200
    assert analysis.json()["profile"] == {
        "userId": 10293,
        "summary": "캠핑을 즐기고 직접 장비를 고르는 분입니다.",
        "keywords": {"taste": [], "interest": [{"value": "캠핑", "score": 0.9}]},
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
                "goalAssessment": {"goal": "INTEREST", "status": "found"},
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
                "goalAssessment": {"goal": "INTEREST", "status": "found"},
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
        {"value": "산책", "score": 0.95},
        {"value": "사진", "score": 0.8},
        {"value": "음악", "score": 0.6},
    ]
    assert updated.status_code == 200
    assert updated.json()["profile"]["keywords"]["interest"] == [
        {"value": "산책", "score": 0.95},
        {"value": "음악", "score": 0.6},
    ]
