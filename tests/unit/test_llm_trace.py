import asyncio
import json

import httpx

from app.application.llm_trace import bind_llm_session, current_llm_trace, llm_call
from app.infrastructure.model_gateway import (
    OpenAICompatibleModelGateway,
    SystemOneJudgmentGateway,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1"
RUNPOD_URL = "https://pod-8000.proxy.runpod.net/v1"
TRACE_FIELDS = {
    "session_id": "42",
    "user": "7",
    "trace": {"trace_name": "reply", "generation_name": "reply", "environment": "test"},
}


def _client(requests: list[dict[str, object]]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"안녕"}}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "안녕"}}], "answers": {}},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _complete_and_stream(base_url: str, requests: list[dict[str, object]]) -> None:
    bind_llm_session(42, user_id=7)
    async with _client(requests) as client:
        gateway = OpenAICompatibleModelGateway(
            client=client,
            model_base_urls={"m": base_url},
            trace_environment="test",
        )
        with llm_call("reply"):
            await gateway.complete([{"role": "user", "content": "안녕"}], model="m")
            async for _ in gateway.stream([{"role": "user", "content": "안녕"}], model="m"):
                pass


def test_openrouter_requests_carry_trace_fields() -> None:
    requests: list[dict[str, object]] = []

    asyncio.run(_complete_and_stream(OPENROUTER_URL, requests))

    assert len(requests) == 2
    for payload in requests:
        assert TRACE_FIELDS.items() <= payload.items()


def test_other_endpoints_never_receive_trace_fields() -> None:
    requests: list[dict[str, object]] = []

    asyncio.run(_complete_and_stream(RUNPOD_URL, requests))

    assert len(requests) == 2
    for payload in requests:
        assert not TRACE_FIELDS.keys() & payload.keys()


def test_judgment_requests_carry_trace_fields() -> None:
    requests: list[dict[str, object]] = []

    async def exercise() -> None:
        bind_llm_session(42, user_id=7)
        async with _client(requests) as client:
            gateway = SystemOneJudgmentGateway(
                client=client,
                base_url=OPENROUTER_URL,
                trace_environment="test",
            )
            with llm_call("reply"):
                await gateway.decide({}, {"q": {"type": "noul"}}, model="jev")

    asyncio.run(exercise())

    assert TRACE_FIELDS.items() <= requests[0].items()


def test_call_name_is_scoped_and_session_reaches_child_tasks() -> None:
    async def exercise() -> None:
        bind_llm_session(42, user_id=7)
        with llm_call("reply"):
            assert current_llm_trace().name == "reply"
        assert current_llm_trace().name is None

        async def child() -> tuple[str | None, str | None]:
            with llm_call("extraction"):
                return current_llm_trace().session_id, current_llm_trace().name

        assert await asyncio.create_task(child()) == ("42", "extraction")
        assert current_llm_trace().name is None

    asyncio.run(exercise())


def test_call_name_survives_stream_closed_early() -> None:
    async def tokens():
        with llm_call("reply"):
            yield "안"
            yield "녕"

    async def exercise() -> None:
        stream = tokens()
        assert await anext(stream) == "안"
        assert current_llm_trace().name == "reply"
        await stream.aclose()
        assert current_llm_trace().name is None

    asyncio.run(exercise())
