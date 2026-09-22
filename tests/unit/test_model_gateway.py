import asyncio
import json

import httpx

from app.infrastructure.model_gateway import OpenAICompatibleModelGateway


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
