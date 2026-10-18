from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.application.ports.model_gateway import Message, ModelGateway
from app.infrastructure.model_gateway.openai_compatible import ModelGatewayError

logger = logging.getLogger(__name__)

# 요청 자체가 잘못된 경우다. 폴백으로 보내도 고쳐지지 않고, 1차 서버가 죽었다는 뜻도 아니다.
_REQUEST_FAULT_STATUSES = frozenset({400, 413, 422})


@dataclass(frozen=True)
class FallbackRoute:
    gateway: ModelGateway
    model: str


def is_unavailable(error: ModelGatewayError) -> bool:
    """1차 서버에 닿지 못했거나 서버가 응답할 수 없는 상태인지 본다.

    꺼진 RunPod Pod는 프록시가 곧바로 404를 준다. 응답 형식 오류(invalid_json 등)는
    서버가 살아 있다는 뜻이므로 넘기지 않고 호출한 쪽의 재시도에 맡긴다.
    """
    if error.reason in {"timeout", "network", "unavailable"}:
        return True
    return error.reason == "http_status" and error.status_code not in _REQUEST_FAULT_STATUSES


class FailoverModelGateway:
    """1차 엔드포인트(RunPod vLLM)가 실패하면 모델별 폴백(OpenRouter)으로 넘긴다.

    시각이 아니라 실패를 기준으로 넘긴다. 한 번 실패하면 cooldown 동안 1차를 건너뛰고,
    cooldown이 지나면 다음 요청이 1차를 다시 시도해 복구 여부를 확인한다.
    GPU를 밤새 꺼 두는 운영이라 실패가 이어지면 cooldown을 두 배씩 max_cooldown까지 늘리고,
    로그는 꺼질 때와 돌아올 때만 남긴다. 폴백이 없는 모델(문서 생성 등)은 1차만 쓴다.
    primary_ready가 있으면(RunPod Pod 조회) 준비되지 않은 동안은 1차를 부르지 않고 바로 넘긴다.
    """

    def __init__(
        self,
        *,
        primary: ModelGateway,
        fallbacks: Mapping[str, FallbackRoute],
        cooldown_seconds: float = 30.0,
        max_cooldown_seconds: float = 120.0,
        primary_ready: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._primary = primary
        self._fallbacks = dict(fallbacks)
        self._primary_ready = primary_ready
        self._cooldown_seconds = cooldown_seconds
        self._max_cooldown_seconds = max(max_cooldown_seconds, cooldown_seconds)
        self._clock = clock
        self._primary_down_until: float | None = None
        self._down_since: float | None = None
        self._failed_probes = 0
        self._fallback_requests = 0

    def _skip_primary(self, model: str) -> bool:
        if model not in self._fallbacks:
            return False
        if self._primary_ready is not None and not self._primary_ready():
            return True
        if self._primary_down_until is None:
            return False
        return self._clock() < self._primary_down_until

    def _mark_down(self, model: str, error: ModelGatewayError) -> None:
        now = self._clock()
        if self._down_since is None:
            self._down_since = now
            logger.warning(
                "primary model unavailable, using fallback until it recovers: "
                "model=%s fallback=%s %s",
                model,
                self._fallbacks[model].model,
                error.describe(),
            )
        else:
            logger.debug("primary model still unavailable: model=%s %s", model, error.describe())
        cooldown = min(self._cooldown_seconds * 2**self._failed_probes, self._max_cooldown_seconds)
        self._failed_probes += 1
        self._primary_down_until = now + cooldown

    def _mark_up(self, model: str) -> None:
        if self._down_since is not None:
            logger.info(
                "primary model recovered: model=%s down_s=%d fallback_requests=%d",
                model,
                self._clock() - self._down_since,
                self._fallback_requests,
            )
        self._primary_down_until = None
        self._down_since = None
        self._failed_probes = 0
        self._fallback_requests = 0

    def _fallback(self, model: str) -> FallbackRoute:
        self._fallback_requests += 1
        return self._fallbacks[model]

    def _should_fail_over(self, model: str, error: ModelGatewayError) -> bool:
        if model not in self._fallbacks or not is_unavailable(error):
            return False
        self._mark_down(model, error)
        return True

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        if not self._skip_primary(model):
            try:
                result = await self._primary.complete(messages, model=model)
            except ModelGatewayError as error:
                if not self._should_fail_over(model, error):
                    raise
            else:
                self._mark_up(model)
                return result
        route = self._fallback(model)
        return await route.gateway.complete(messages, model=route.model)

    async def structured(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self._skip_primary(model):
            try:
                result = await self._primary.structured(
                    messages, model=model, json_schema=json_schema
                )
            except ModelGatewayError as error:
                if not self._should_fail_over(model, error):
                    raise
            else:
                self._mark_up(model)
                return result
        route = self._fallback(model)
        return await route.gateway.structured(messages, model=route.model, json_schema=json_schema)

    async def stream(self, messages: Sequence[Message], *, model: str) -> AsyncIterator[str]:
        if not self._skip_primary(model):
            started = False
            try:
                async for text in self._primary.stream(messages, model=model):
                    if not started:
                        started = True
                        self._mark_up(model)
                    yield text
                return
            except ModelGatewayError as error:
                # 이미 내보낸 글자는 되돌릴 수 없으므로 첫 토큰 전에 실패한 경우만 넘긴다.
                if started or not self._should_fail_over(model, error):
                    raise
        route = self._fallback(model)
        async for text in route.gateway.stream(messages, model=route.model):
            yield text
