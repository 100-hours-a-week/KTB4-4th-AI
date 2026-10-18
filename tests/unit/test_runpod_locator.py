import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.infrastructure.model_gateway import FailoverModelGateway, RunPodPodLocator
from app.main import _build_model_gateway, _build_pod_locator

MESSAGES = [{"role": "user", "content": "안녕"}]


def pod(pod_id: str, name: str, status: str, created_at: str) -> dict[str, Any]:
    return {
        "id": pod_id,
        "name": name,
        "status": status,
        "ports": ["8000/http"],
        "createdAt": created_at,
    }


class FakeRunPod:
    """RunPod API와 각 Pod의 vLLM /health를 흉내 낸다."""

    def __init__(self) -> None:
        self.pods: list[dict[str, Any]] = []
        self.healthy: set[str] = set()
        self.api_down = False
        self.chat_hosts: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.runpod.io":
            assert request.headers["Authorization"] == "Bearer runpod-read-key"
            if self.api_down:
                return httpx.Response(503)
            return httpx.Response(200, json={"pods": self.pods})
        if host.endswith(".proxy.runpod.net"):
            pod_id = host.removesuffix("-8000.proxy.runpod.net")
            if request.url.path == "/health":
                return httpx.Response(200 if pod_id in self.healthy else 404)
            self.chat_hosts.append(pod_id)
            return httpx.Response(200, json={"choices": [{"message": {"content": "kanana"}}]})
        if host == "openrouter.ai":
            self.chat_hosts.append("openrouter")
            return httpx.Response(200, json={"choices": [{"message": {"content": "haiku"}}]})
        raise AssertionError(f"unexpected request: {request.url}")


def locator(client: httpx.AsyncClient) -> RunPodPodLocator:
    return RunPodPodLocator(
        client=client, api_key="runpod-read-key", name_prefix="needu-kanana-vllm"
    )


def run_with(fake: FakeRunPod, scenario) -> Any:
    async def exercise() -> Any:
        async with httpx.AsyncClient(
            timeout=None, transport=httpx.MockTransport(fake.handler)
        ) as c:
            return await scenario(c)

    return asyncio.run(exercise())


def test_finds_newest_running_pod_by_prefix_after_migration() -> None:
    fake = FakeRunPod()
    fake.pods = [
        pod("old0001", "needu-kanana-vllm", "EXITED", "2026-10-16T02:09:57Z"),
        pod("new0002", "needu-kanana-vllm-migration", "RUNNING", "2026-10-16T06:51:45Z"),
        pod("other03", "someone-else", "RUNNING", "2026-10-16T07:00:00Z"),
    ]
    fake.healthy = {"new0002", "other03"}

    async def scenario(client: httpx.AsyncClient) -> str | None:
        found = locator(client)
        await found.refresh()
        return found.base_url

    assert run_with(fake, scenario) == "https://new0002-8000.proxy.runpod.net/v1"


def test_running_pod_still_loading_model_is_not_ready() -> None:
    fake = FakeRunPod()
    fake.pods = [pod("boot001", "needu-kanana-vllm", "RUNNING", "2026-10-16T09:00:00Z")]

    async def scenario(client: httpx.AsyncClient) -> list[str | None]:
        found = locator(client)
        await found.refresh()
        before = found.base_url
        fake.healthy.add("boot001")  # vLLM이 모델을 다 올렸다
        await found.refresh()
        return [before, found.base_url]

    assert run_with(fake, scenario) == [None, "https://boot001-8000.proxy.runpod.net/v1"]


def test_lookup_failure_keeps_last_endpoint_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeRunPod()
    fake.pods = [pod("pod0001", "needu-kanana-vllm", "RUNNING", "2026-10-16T09:00:00Z")]
    fake.healthy = {"pod0001"}
    caplog.set_level(logging.WARNING, logger="app.infrastructure.model_gateway.runpod_locator")

    async def scenario(client: httpx.AsyncClient) -> str | None:
        found = locator(client)
        await found.refresh()
        fake.api_down = True
        for _ in range(5):
            await found.refresh()
        return found.base_url

    assert run_with(fake, scenario) == "https://pod0001-8000.proxy.runpod.net/v1"
    assert len(caplog.records) == 1


def runpod_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "response_model_name": "kanana-response",
        "extraction_model_name": "kanana-extractor",
        "response_fallback_model_base_url": "https://openrouter.ai/api/v1",
        "response_fallback_model_name": "anthropic/claude-haiku-5.5",
        "extraction_fallback_model_base_url": "https://openrouter.ai/api/v1",
        "extraction_fallback_model_name": "anthropic/claude-haiku-5.5",
        "model_api_key": "vllm-key",
        "openrouter_api_key": "openrouter-key",
        "runpod_api_key": "runpod-read-key",
        "runpod_pod_name_prefix": "needu-kanana-vllm",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_gateway_follows_pod_and_skips_it_while_off() -> None:
    fake = FakeRunPod()

    async def scenario(client: httpx.AsyncClient) -> list[str]:
        settings = runpod_settings()
        found = _build_pod_locator(settings, client)
        gateway = _build_model_gateway(
            settings, client, api_key="vllm-key", fallback_api_key="openrouter-key", locator=found
        )
        assert isinstance(gateway, FailoverModelGateway)
        replies = []
        await found.refresh()  # Pod가 하나도 없다
        replies.append(await gateway.complete(MESSAGES, model="kanana-response"))
        fake.pods = [pod("mig0001", "needu-kanana-vllm-migration", "RUNNING", "2026-10-16T09:00Z")]
        fake.healthy = {"mig0001"}
        await found.refresh()  # 아침에 켰다
        replies.append(await gateway.complete(MESSAGES, model="kanana-response"))
        return replies

    assert run_with(fake, scenario) == ["haiku", "kanana"]
    # 꺼져 있는 동안에는 Pod 주소로 요청을 보내지 않는다.
    assert fake.chat_hosts == ["openrouter", "mig0001"]


def test_prefix_without_runpod_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        runpod_settings(runpod_api_key=None)


def test_chat_request_carries_vllm_key_not_runpod_key() -> None:
    seen: list[str] = []
    fake = FakeRunPod()
    fake.pods = [pod("pod0001", "needu-kanana-vllm", "RUNNING", "2026-10-16T09:00:00Z")]
    fake.healthy = {"pod0001"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            seen.append(request.headers["Authorization"])
            assert json.loads(request.content)["model"] == "kanana-extractor"
        return fake.handler(request)

    async def exercise() -> None:
        async with httpx.AsyncClient(timeout=None, transport=httpx.MockTransport(handler)) as c:
            settings = runpod_settings()
            found = _build_pod_locator(settings, c)
            await found.refresh()
            gateway = _build_model_gateway(
                settings, c, api_key="vllm-key", fallback_api_key="openrouter-key", locator=found
            )
            await gateway.complete(MESSAGES, model="kanana-extractor")

    asyncio.run(exercise())
    assert seen == ["Bearer vllm-key"]
