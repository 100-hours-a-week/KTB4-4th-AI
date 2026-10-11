import asyncio
import json

import httpx

from app.infrastructure.model_gateway import (
    ModelGatewayError,
    OpenAICompatibleModelGateway,
    SystemOneJudgmentGateway,
)


def test_openai_compatible_gateway_supports_completion_and_structured_output() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if "response_format" in payload:
            content = json.dumps({"items": [], "goalAssessment": {"status": "unresolved"}})
        else:
            content = "안녕하세요"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    async def exercise() -> tuple[str, dict[str, object]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = OpenAICompatibleModelGateway(
                client=client,
                model_base_urls={"local-model": "http://localhost:1234/v1/"},
                api_key="test-key",
            )
            completion = await gateway.complete(
                [{"role": "user", "content": "인사해줘"}],
                model="local-model",
            )
            structured = await gateway.structured(
                [{"role": "user", "content": "추출해줘"}],
                model="local-model",
                json_schema={"type": "object"},
            )
        return completion, dict(structured)

    completion, structured = asyncio.run(exercise())

    assert completion == "안녕하세요"
    assert structured["items"] == []
    assert requests[0]["model"] == "local-model"
    assert requests[1]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_response",
            "strict": True,
            "schema": {"type": "object"},
        },
    }


def test_sends_configured_max_tokens_per_model() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = OpenAICompatibleModelGateway(
                client=client,
                model_base_urls={"capped": "http://x/v1", "uncapped": "http://x/v1"},
                model_max_tokens={"capped": 256},
            )
            await gateway.complete([{"role": "user", "content": "hi"}], model="capped")
            await gateway.complete([{"role": "user", "content": "hi"}], model="uncapped")

    asyncio.run(exercise())

    assert payloads[0]["max_tokens"] == 256
    assert "max_tokens" not in payloads[1]


def test_sends_configured_thinking_mode_per_model() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = OpenAICompatibleModelGateway(
                client=client,
                model_base_urls={"gemma": "http://x/v1", "other": "http://x/v1"},
                model_enable_thinking={"gemma": False},
            )
            await gateway.complete([{"role": "user", "content": "hi"}], model="gemma")
            await gateway.complete([{"role": "user", "content": "hi"}], model="other")

    asyncio.run(exercise())

    assert payloads[0]["enable_thinking"] is False
    assert "enable_thinking" not in payloads[1]


def test_sends_configured_temperature_and_stop_sequences_per_model() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = OpenAICompatibleModelGateway(
                client=client,
                model_base_urls={"kanana": "http://x/v1", "other": "http://x/v1"},
                model_temperatures={"kanana": 0.2},
                model_stop_sequences={"kanana": ("<|eot_id|>",)},
            )
            await gateway.complete([{"role": "user", "content": "hi"}], model="kanana")
            await gateway.complete([{"role": "user", "content": "hi"}], model="other")

    asyncio.run(exercise())

    assert payloads[0]["temperature"] == 0.2
    assert payloads[0]["stop"] == ["<|eot_id|>"]
    assert "temperature" not in payloads[1]
    assert "stop" not in payloads[1]


def test_judgment_gateway_posts_state_and_questions_to_systemone() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13",
                "answers": {"q": {"type": "noul", "noul": 0.9}},
            },
        )

    async def exercise() -> dict[str, object]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = SystemOneJudgmentGateway(
                client=client,
                base_url="https://openrouter.ai/api/v1/",
                api_key="test-key",
            )
            return dict(
                await gateway.decide(
                    {"user_message": "러닝은 싫어요"},
                    {"q": {"type": "noul", "instructions": "Is this negative?"}},
                    model="~typesafe/jev-latest",
                )
            )

    body = asyncio.run(exercise())

    assert body["answers"] == {"q": {"type": "noul", "noul": 0.9}}
    assert str(requests[0].url) == "https://openrouter.ai/api/v1/systemone"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {
        "model": "~typesafe/jev-latest",
        "state": {"user_message": "러닝은 싫어요"},
        "questions": {"q": {"type": "noul", "instructions": "Is this negative?"}},
    }


def _failure(handler) -> ModelGatewayError:
    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = OpenAICompatibleModelGateway(
                client=client, model_base_urls={"m": "http://localhost/v1"}
            )
            await gateway.complete([{"role": "user", "content": "hi"}], model="m")

    try:
        asyncio.run(exercise())
    except ModelGatewayError as error:
        return error
    raise AssertionError("expected ModelGatewayError")


def test_http_error_keeps_status_provider_and_upstream_message() -> None:
    error = _failure(
        lambda request: httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "Google", "raw": "RESOURCE_EXHAUSTED"},
                }
            },
        )
    )

    assert error.reason == "http_status"
    assert error.status_code == 429
    assert error.provider == "Google"
    assert "RESOURCE_EXHAUSTED" in error.describe()


def test_error_body_with_200_is_upstream_error() -> None:
    error = _failure(
        lambda request: httpx.Response(
            200,
            json={"error": {"message": "Internal error", "metadata": {"provider_name": "Google"}}},
        )
    )

    assert error.reason == "upstream_error"
    assert error.provider == "Google"


def test_empty_content_reports_finish_reason() -> None:
    error = _failure(
        lambda request: httpx.Response(
            200,
            json={
                "provider": "Google",
                "choices": [{"finish_reason": "length", "message": {"content": ""}}],
            },
        )
    )

    assert error.reason == "empty_content"
    assert error.detail == "finish_reason=length"


def test_transport_failures_are_timeout_or_network() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _failure(timeout).reason == "timeout"
    assert _failure(refused).reason == "network"
