from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RunPodPodLocator:
    """RunPod API로 이름 접두어가 맞는 Pod를 찾아 응답·추출 1차 주소를 정한다.

    Pod는 머신에 묶여 있어서 켤 때 자리가 없으면 새 Pod(새 ID·새 URL)로 옮겨야 한다.
    마이그레이션한 Pod는 이름에 접미어가 붙으므로 정확한 이름이 아니라 접두어로 찾는다.
    RUNNING은 컨테이너가 떴다는 뜻일 뿐이라 vLLM /health가 200일 때만 주소를 내준다.
    조회가 실패하면 마지막으로 확인한 주소를 유지하고, 그 주소가 죽었으면 폴백이 받는다.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        name_prefix: str,
        port: int = 8000,
        api_base_url: str = "https://api.runpod.io",
        interval_seconds: float = 30.0,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._name_prefix = name_prefix
        self._port = port
        self._pods_url = f"{api_base_url.rstrip('/')}/v2/pods"
        self._interval_seconds = interval_seconds
        self._timeout = httpx.Timeout(request_timeout_seconds)
        self._base_url: str | None = None
        self._lookup_failing = False
        self._checked = False

    @property
    def base_url(self) -> str | None:
        """준비된 Pod의 OpenAI 호환 주소(…/v1). 없으면 None이다."""
        return self._base_url

    def is_ready(self) -> bool:
        return self._base_url is not None

    def _pick(self, pods: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        running = [
            pod
            for pod in pods
            if str(pod.get("name", "")).startswith(self._name_prefix)
            and pod.get("status") == "RUNNING"
        ]
        # 마이그레이션 직후 옛 Pod가 남아 있어도 가장 최근에 만든 Pod를 쓴다.
        return max(running, key=lambda pod: str(pod.get("createdAt", "")), default=None)

    async def _list_pods(self) -> list[Mapping[str, Any]]:
        response = await self._client.get(
            self._pods_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()
        pods = body.get("pods") if isinstance(body, Mapping) else None
        if not isinstance(pods, list):
            raise ValueError("RunPod pods response has no pods list")
        return pods

    async def _healthy(self, root: str) -> bool:
        try:
            response = await self._client.get(f"{root}/health", timeout=self._timeout)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def refresh(self) -> None:
        try:
            pods = await self._list_pods()
        except (httpx.HTTPError, ValueError) as error:
            if not self._lookup_failing:
                logger.warning(
                    "runpod pod lookup failed, keeping last endpoint=%s: %s: %s",
                    self._base_url,
                    type(error).__name__,
                    error,
                )
            self._lookup_failing = True
            return
        self._lookup_failing = False

        pod = self._pick(pods)
        root = f"https://{pod['id']}-{self._port}.proxy.runpod.net" if pod else None
        base_url = f"{root}/v1" if root and await self._healthy(root) else None
        # 상태가 바뀔 때만 남기되, 서버를 띄운 직후 첫 결과는 꺼져 있어도 한 번 남긴다.
        if base_url != self._base_url or not self._checked:
            self._checked = True
            if base_url is None:
                logger.warning(
                    "runpod pod not ready, using fallback: prefix=%s pod=%s status=%s",
                    self._name_prefix,
                    pod["id"] if pod else None,
                    "starting" if pod else "no running pod",
                )
            else:
                logger.info("runpod pod ready: id=%s name=%s", pod["id"], pod.get("name"))
            self._base_url = base_url

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_seconds)
            try:
                await self.refresh()
            except Exception:
                # 조회 루프가 죽으면 주소가 다시는 바뀌지 않으므로 로그만 남기고 계속 돈다.
                logger.exception("runpod pod lookup crashed")
