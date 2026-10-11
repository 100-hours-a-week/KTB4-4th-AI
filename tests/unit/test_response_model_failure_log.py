import asyncio
import logging

import pytest

from app.application.chat_use_cases import ChatUseCases, ResponseModelUnavailableError
from app.application.conversation_service import ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.infrastructure.model_gateway import ModelGatewayError
from app.infrastructure.persistence import InMemorySessionStore


class FailingGateway:
    def __init__(self, error: Exception | None = None, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay

    async def complete(self, messages, *, model: str) -> str:
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return "안녕하세요"


def use_cases(gateway: FailingGateway, *, timeout: float = 5.0) -> ChatUseCases:
    return ChatUseCases(
        conversations=ConversationService(model_gateway=gateway, extraction_model="x"),
        states=ConversationStateStore(InMemorySessionStore()),
        model_gateway=gateway,
        response_model="google/gemini",
        response_timeout_seconds=timeout,
    )


def test_gateway_failure_is_logged_with_reason_and_upstream_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ModelGatewayError(
        "Model endpoint request failed",
        reason="http_status",
        status_code=429,
        provider="Google",
        detail="RESOURCE_EXHAUSTED",
    )
    with caplog.at_level(logging.WARNING), pytest.raises(ResponseModelUnavailableError):
        asyncio.run(
            use_cases(FailingGateway(error)).create_session(user_id=1, conversation_room_id=7)
        )

    record = caplog.records[-1]
    message = record.getMessage()
    assert "purpose=opening session=7 model=google/gemini" in message
    assert "reason=http_status status=429 provider=Google detail=RESOURCE_EXHAUSTED" in message
    assert record.exc_info is not None


def test_timeout_is_logged_as_our_timeout(caplog: pytest.LogCaptureFixture) -> None:
    gateway = FailingGateway(delay=0.2)
    with caplog.at_level(logging.WARNING), pytest.raises(ResponseModelUnavailableError):
        asyncio.run(
            use_cases(gateway, timeout=0.05).create_session(user_id=1, conversation_room_id=8)
        )

    assert "reason=timeout limit_s=0.05" in caplog.records[-1].getMessage()
