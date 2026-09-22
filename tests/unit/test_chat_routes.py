from collections.abc import AsyncIterator, Mapping, Sequence
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


def test_chat_http_lifecycle_matches_v1_contract() -> None:
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
            json={"message": "주말마다 캠핑 가요"},
        )
        analysis = client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        closed = client.post("/v1/chat/sessions/45678/close", headers=headers)
        closed_again = client.post("/v1/chat/sessions/45678/close", headers=headers)
        deleted = client.delete("/v1/chat/sessions/45678", headers=headers)

    assert created.status_code == 201
    assert created.json() == {
        "sessionId": 45678,
        "greeting": "안녕하세요! 요즘 어떻게 지내세요?",
        "maxTurns": 20,
    }
    assert message.status_code == 200
    assert message.json() == {
        "reply": "캠핑 좋죠. 주로 어디로 다니세요?",
        "turn": 1,
        "maxTurns": 20,
        "canClose": False,
        "itemCount": 1,
        "inputLocked": False,
        "completionReason": None,
        "profileCompleteness": None,
        "lastTurnExtractionFailed": False,
    }
    assert analysis.status_code == 200
    assert analysis.json()["profile"]["hobbies"][0]["value"] == "캠핑"
    assert analysis.json()["summary"] == "캠핑을 즐기고 직접 장비를 고르는 분입니다."
    assert analysis.json()["keywords"] == {"taste": [], "interest": ["캠핑"]}
    assert analysis.json()["correctionAvailable"] is True
    assert closed.status_code == 200
    assert closed.json()["profile"] == analysis.json()["profile"]
    assert closed.json()["summary"] == analysis.json()["summary"]
    assert "correctionAvailable" not in closed.json()
    assert closed_again.status_code == 409
    assert closed_again.json()["code"] == "SESSION_CLOSED"
    assert deleted.status_code == 204
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


def test_get_chat_session_restores_history_and_current_state() -> None:
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=FakeModelGateway(),
        session_store=InMemorySessionStore(),
    )
    headers = {"Authorization": "Bearer test-token"}

    with TestClient(app) as client:
        missing = client.get("/v1/chat/sessions/99999?userId=10293", headers=headers)
        client.post(
            "/v1/chat/sessions",
            headers=headers,
            json={"userId": 10293, "conversationRoomId": 45678},
        )
        client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"message": "주말마다 캠핑 가요"},
        )
        wrong_user = client.get("/v1/chat/sessions/45678?userId=77777", headers=headers)
        active = client.get("/v1/chat/sessions/45678?userId=10293", headers=headers)
        client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        review = client.get("/v1/chat/sessions/45678?userId=10293", headers=headers)
        client.post("/v1/chat/sessions/45678/close", headers=headers)
        closed = client.get("/v1/chat/sessions/45678?userId=10293", headers=headers)

    assert missing.status_code == 404
    assert missing.json()["code"] == "SESSION_NOT_FOUND"
    assert wrong_user.status_code == 404
    assert wrong_user.json()["code"] == "SESSION_NOT_FOUND"
    assert active.status_code == 200
    assert active.json()["sessionId"] == 45678
    assert active.json()["userId"] == 10293
    assert active.json()["status"] == "active"
    assert active.json()["turn"] == 1
    assert active.json()["itemCount"] == 1
    assert active.json()["inputLocked"] is False
    assert active.json()["analysisAvailable"] is False
    assert active.json()["messages"] == [
        {"role": "assistant", "content": "안녕하세요! 요즘 어떻게 지내세요?"},
        {"role": "user", "content": "주말마다 캠핑 가요"},
        {"role": "assistant", "content": "캠핑 좋죠. 주로 어디로 다니세요?"},
    ]
    assert review.status_code == 200
    assert review.json()["status"] == "review"
    assert review.json()["analysisAvailable"] is True
    assert review.json()["profileCompleteness"] == "partial"
    assert closed.status_code == 409
    assert closed.json()["code"] == "SESSION_CLOSED"


def test_close_requires_current_analysis_and_correction_reuses_message_api() -> None:
    gateway = FakeModelGateway(
        completions=[
            "안녕하세요! 요즘 어떻게 지내세요?",
            "캠핑을 좋아하시는군요.",
            "캠핑을 즐기는 분입니다.",
            "캠핑이 아니라 등산이라는 내용으로 바로잡아 둘게요.",
            "등산을 즐기는 분입니다.",
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
            {
                "items": [
                    {
                        "field": "hobbies",
                        "value": "등산",
                        "confidence": 0.9,
                        "evidence": "등산을 좋아해요",
                        "evidenceType": "explicit",
                        "intentType": "both",
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [{"field": "hobbies", "value": "캠핑"}],
                "goalAssessment": {"goal": "CORRECT", "status": "found"},
            },
        ],
    )
    app = create_app(
        Settings(app_env="test", service_token="test-token", redis_url=None),
        model_gateway=gateway,
        session_store=InMemorySessionStore(),
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
            json={"message": "캠핑을 좋아해요"},
        )
        close_without_analysis = client.post("/v1/chat/sessions/45678/close", headers=headers)
        client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        correction = client.post(
            "/v1/chat/sessions/45678/messages",
            headers=headers,
            json={"message": "캠핑이 아니라 등산을 좋아해요"},
        )
        close_with_stale_analysis = client.post("/v1/chat/sessions/45678/close", headers=headers)
        corrected_analysis = client.post("/v1/chat/sessions/45678/analysis", headers=headers)
        closed = client.post("/v1/chat/sessions/45678/close", headers=headers)

    assert close_without_analysis.status_code == 409
    assert close_without_analysis.json()["code"] == "ANALYSIS_REQUIRED"
    assert correction.status_code == 200
    assert close_with_stale_analysis.status_code == 409
    assert close_with_stale_analysis.json()["code"] == "ANALYSIS_OUTDATED"
    assert corrected_analysis.json()["profile"]["hobbies"][0]["value"] == "등산"
    assert closed.status_code == 200
    assert closed.json()["profile"]["hobbies"][0]["value"] == "등산"

