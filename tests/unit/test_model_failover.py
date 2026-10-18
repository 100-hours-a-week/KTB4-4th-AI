import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as main_module
from app.core.config import Settings
from app.infrastructure.model_gateway import (
    FailoverModelGateway,
    FallbackRoute,
    ModelGatewayError,
)
from app.main import _build_model_gateway, create_app

MESSAGES = [{"role": "user", "content": "선물 추천해줘"}]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeGateway:
    def __init__(
        self,
        name: str,
        *,
        error: ModelGatewayError | None = None,
        tokens: Sequence[str] = ("안", "녕"),
        fail_after_tokens: int | None = None,
    ) -> None:
        self.name = name
        self.error = error
        self.tokens = tokens
        self.fail_after_tokens = fail_after_tokens
        self.calls: list[tuple[str, str]] = []

    async def complete(self, messages: Sequence[Mapping[str, str]], *, model: str) -> str:
        self.calls.append(("complete", model))
        if self.error is not None:
            raise self.error
        return f"{self.name}:{model}"

    async def structured(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append(("structured", model))
        if self.error is not None:
            raise self.error
        return {"by": self.name, "schema": dict(json_schema)}

    async def stream(
        self, messages: Sequence[Mapping[str, str]], *, model: str
    ) -> AsyncIterator[str]:
        self.calls.append(("stream", model))
        for index, token in enumerate(self.tokens):
            if self.fail_after_tokens is not None and index == self.fail_after_tokens:
                raise ModelGatewayError("cut", reason="network")
            yield token
        if self.error is not None and self.fail_after_tokens is None:
            raise self.error


def pod_off() -> ModelGatewayError:
    # 꺼진 RunPod Pod는 프록시가 곧바로 404를 준다.
    return ModelGatewayError("down", reason="http_status", status_code=404)


def failover(
    primary: FakeGateway, fallback: FakeGateway, clock: FakeClock | None = None
) -> FailoverModelGateway:
    return FailoverModelGateway(
        primary=primary,
        fallbacks={
            "kanana-response": FallbackRoute(gateway=fallback, model="anthropic/haiku"),
            "kanana-extractor": FallbackRoute(gateway=fallback, model="anthropic/haiku"),
        },
        cooldown_seconds=30,
        clock=clock or FakeClock(),
    )


def test_pod_off_falls_back_with_fallback_model_name() -> None:
    primary, fallback = FakeGateway("runpod", error=pod_off()), FakeGateway("openrouter")
    gateway = failover(primary, fallback)

    result = asyncio.run(gateway.complete(MESSAGES, model="kanana-response"))

    assert result == "openrouter:anthropic/haiku"
    assert primary.calls == [("complete", "kanana-response")]


def test_structured_falls_back_with_same_schema() -> None:
    primary = FakeGateway("runpod", error=ModelGatewayError("t", reason="timeout"))
    fallback = FakeGateway("openrouter")
    gateway = failover(primary, fallback)

    result = asyncio.run(
        gateway.structured(MESSAGES, model="kanana-extractor", json_schema={"type": "object"})
    )

    assert result == {"by": "openrouter", "schema": {"type": "object"}}


def test_cooldown_skips_primary_then_retries_and_recovers() -> None:
    clock = FakeClock()
    primary, fallback = FakeGateway("runpod", error=pod_off()), FakeGateway("openrouter")
    gateway = failover(primary, fallback, clock)

    asyncio.run(gateway.complete(MESSAGES, model="kanana-response"))
    clock.now += 10
    asyncio.run(gateway.structured(MESSAGES, model="kanana-extractor", json_schema={}))
    assert len(primary.calls) == 1  # cooldown 동안은 1차를 부르지 않는다

    primary.error = None  # Pod를 다시 켰다
    clock.now += 30
    assert asyncio.run(gateway.complete(MESSAGES, model="kanana-response")) == (
        "runpod:kanana-response"
    )
    clock.now += 1
    assert asyncio.run(gateway.complete(MESSAGES, model="kanana-response")) == (
        "runpod:kanana-response"
    )
    assert len(primary.calls) == 3


def test_gpu_off_overnight_logs_once_and_backs_off(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    primary, fallback = FakeGateway("runpod", error=pod_off()), FakeGateway("openrouter")
    gateway = failover(primary, fallback, clock)
    probe_times: list[float] = []

    caplog.set_level(logging.INFO, logger="app.infrastructure.model_gateway.failover")
    for _ in range(600):  # 10초마다 요청, 100분 동안
        before = len(primary.calls)
        asyncio.run(gateway.complete(MESSAGES, model="kanana-response"))
        if len(primary.calls) > before:
            probe_times.append(clock.now - 1000.0)
        clock.now += 10

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    # 재시도 간격이 30초, 60초, 120초로 늘고 120초에서 멈춘다.
    assert probe_times[:5] == [0, 30, 90, 210, 330]

    primary.error = None  # 아침에 GPU를 켰다
    clock.now += 120
    asyncio.run(gateway.complete(MESSAGES, model="kanana-response"))
    recovered = [r.getMessage() for r in caplog.records if "recovered" in r.getMessage()]
    assert len(recovered) == 1
    assert "fallback_requests=600" in recovered[0]


@pytest.mark.parametrize(
    "error",
    [
        ModelGatewayError("bad", reason="http_status", status_code=400),
        ModelGatewayError("bad", reason="invalid_structured"),
        ModelGatewayError("bad", reason="empty_content"),
    ],
)
def test_request_or_content_errors_are_not_failed_over(error: ModelGatewayError) -> None:
    primary, fallback = FakeGateway("runpod", error=error), FakeGateway("openrouter")
    gateway = failover(primary, fallback)

    with pytest.raises(ModelGatewayError) as raised:
        asyncio.run(gateway.complete(MESSAGES, model="kanana-response"))

    assert raised.value is error
    assert fallback.calls == []


def test_model_without_fallback_keeps_primary_error() -> None:
    primary, fallback = FakeGateway("runpod", error=pod_off()), FakeGateway("openrouter")
    gateway = failover(primary, fallback)

    with pytest.raises(ModelGatewayError):
        asyncio.run(gateway.complete(MESSAGES, model="document-model"))
    assert fallback.calls == []


async def collect(stream: AsyncIterator[str]) -> list[str]:
    return [token async for token in stream]


def test_stream_falls_back_when_primary_fails_before_first_token() -> None:
    primary = FakeGateway("runpod", error=pod_off(), tokens=())
    fallback = FakeGateway("openrouter", tokens=("폴", "백"))
    gateway = failover(primary, fallback)

    tokens = asyncio.run(collect(gateway.stream(MESSAGES, model="kanana-response")))

    assert tokens == ["폴", "백"]
    assert fallback.calls == [("stream", "anthropic/haiku")]


def test_stream_does_not_switch_models_after_tokens_were_sent() -> None:
    primary = FakeGateway("runpod", tokens=("안", "녕"), fail_after_tokens=1)
    fallback = FakeGateway("openrouter")
    gateway = failover(primary, fallback)
    received: list[str] = []

    async def consume() -> None:
        async for token in gateway.stream(MESSAGES, model="kanana-response"):
            received.append(token)

    with pytest.raises(ModelGatewayError):
        asyncio.run(consume())
    assert received == ["안"]
    assert fallback.calls == []


def runpod_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "response_model_base_url": "https://pod-8000.proxy.runpod.net/v1",
        "extraction_model_base_url": "https://pod-8000.proxy.runpod.net/v1",
        "response_model_name": "kanana-response",
        "extraction_model_name": "kanana-extractor",
        "response_fallback_model_base_url": "https://openrouter.ai/api/v1",
        "response_fallback_model_name": "anthropic/claude-haiku-5.5",
        "extraction_fallback_model_base_url": "https://openrouter.ai/api/v1",
        "extraction_fallback_model_name": "anthropic/claude-haiku-5.5",
        "model_api_key": "vllm-key",
        "openrouter_api_key": "openrouter-key",
        "judgment_model_base_url": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_runpod_off_routes_to_openrouter_with_each_key() -> None:
    seen: list[tuple[str, str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(
            (
                request.url.host,
                body["model"],
                request.headers["Authorization"],
                request.extensions.get("timeout", {}).get("connect"),
            )
        )
        if request.url.host.endswith("runpod.net"):
            return httpx.Response(404, text="")
        return httpx.Response(200, json={"choices": [{"message": {"content": "폴백 응답"}}]})

    async def exercise() -> str:
        # 운영과 같이 클라이언트 기본값은 제한 없음이고, 1차 호출에만 제한이 붙는다.
        async with httpx.AsyncClient(
            timeout=None, transport=httpx.MockTransport(handler)
        ) as client:
            gateway = _build_model_gateway(
                runpod_settings(),
                client,
                api_key="vllm-key",
                fallback_api_key="openrouter-key",
            )
            return await gateway.complete(MESSAGES, model="kanana-response")

    assert asyncio.run(exercise()) == "폴백 응답"
    assert seen == [
        ("pod-8000.proxy.runpod.net", "kanana-response", "Bearer vllm-key", 3.0),
        ("openrouter.ai", "anthropic/claude-haiku-5.5", "Bearer openrouter-key", None),
    ]


def test_without_fallback_settings_gateway_is_unchanged() -> None:
    settings = runpod_settings(
        response_fallback_model_base_url=None,
        response_fallback_model_name=None,
        extraction_fallback_model_base_url=None,
        extraction_fallback_model_name=None,
    )

    gateway = _build_model_gateway(
        settings, httpx.AsyncClient(), api_key="vllm-key", fallback_api_key=None
    )

    assert not isinstance(gateway, FailoverModelGateway)


def test_fallback_url_without_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        runpod_settings(response_fallback_model_name=None)


def test_judgment_uses_openrouter_key_while_responses_use_vllm_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys: dict[str, str | None] = {}

    class CapturingJudgment:
        def __init__(self, *, client: httpx.AsyncClient, base_url: str, api_key: str | None):
            keys["judgment"] = api_key

    monkeypatch.setattr(main_module, "SystemOneJudgmentGateway", CapturingJudgment)
    app = create_app(
        runpod_settings(judgment_model_base_url="https://openrouter.ai/api/v1", redis_url=None)
    )

    with TestClient(app):
        pass

    assert keys == {"judgment": "openrouter-key"}
