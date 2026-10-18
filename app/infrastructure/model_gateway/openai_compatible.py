from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

import httpx

from app.application.ports.model_gateway import Message

# 로그에 남기는 상위 응답 본문의 최대 길이.
UPSTREAM_DETAIL_CHARS = 300


class ModelGatewayError(RuntimeError):
    """Raised when an OpenAI-compatible model endpoint cannot produce a valid response.

    reason은 실패를 나눈 이름이다. timeout과 network는 상위 서버에 닿지 못한 것,
    unavailable은 주소를 물어봤는데 지금 떠 있는 서버가 없는 것,
    http_status는 상위가 오류 코드로 답한 것, upstream_error는 200 응답 본문에 오류가 담긴 것,
    invalid_json, no_content, empty_content, invalid_structured는 응답 형식이 틀린 것이다.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        status_code: int | None = None,
        provider: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code
        self.provider = provider
        self.detail = detail

    def describe(self) -> str:
        parts = [f"reason={self.reason}"]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.detail:
            parts.append(f"detail={self.detail}")
        return " ".join(parts)


def _upstream_error(body: object) -> tuple[str | None, str | None]:
    """OpenRouter 오류 본문을 읽는다.

    형식: {"error": {"message", "metadata": {"provider_name", "raw"}}}
    """
    if not isinstance(body, Mapping) or not isinstance(body.get("error"), Mapping):
        return None, None
    error = body["error"]
    metadata = error.get("metadata") if isinstance(error.get("metadata"), Mapping) else {}
    provider = metadata.get("provider_name")
    message = str(error.get("message") or "")
    raw = metadata.get("raw")
    if raw:
        message = f"{message} | raw: {raw}"
    return (str(provider) if provider else None), message[:UPSTREAM_DETAIL_CHARS] or None


def request_error(error: Exception, message: str) -> ModelGatewayError:
    """httpx 예외를 원인별 reason을 가진 ModelGatewayError로 바꾼다."""
    if isinstance(error, httpx.TimeoutException):
        return ModelGatewayError(message, reason="timeout", detail=type(error).__name__)
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        try:
            provider, detail = _upstream_error(response.json())
        except ValueError:
            provider, detail = None, None
        return ModelGatewayError(
            message,
            reason="http_status",
            status_code=response.status_code,
            provider=provider,
            detail=detail or response.text[:UPSTREAM_DETAIL_CHARS] or None,
        )
    if isinstance(error, httpx.TransportError):
        return ModelGatewayError(
            message, reason="network", detail=f"{type(error).__name__}: {error}"
        )
    return ModelGatewayError(message, reason="invalid_json", detail=type(error).__name__)


class OpenAICompatibleModelGateway:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model_base_urls: Mapping[str, str | Callable[[], str | None]],
        api_key: str | None = None,
        model_max_tokens: Mapping[str, int] | None = None,
        model_enable_thinking: Mapping[str, bool] | None = None,
        model_temperatures: Mapping[str, float] | None = None,
        model_stop_sequences: Mapping[str, Sequence[str]] | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._client = client
        # 폴백이 있는 1차 엔드포인트만 짧은 타임아웃을 건다. 없으면 클라이언트 설정을 따른다.
        self._request_options: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        # 주소 대신 함수를 받으면 호출할 때마다 물어본다(RunPod Pod처럼 주소가 바뀌는 경우).
        self._model_base_urls = {
            model: base_url if callable(base_url) else base_url.rstrip("/")
            for model, base_url in model_base_urls.items()
        }
        self._api_key = api_key
        self._model_max_tokens = dict(model_max_tokens or {})
        self._model_enable_thinking = dict(model_enable_thinking or {})
        self._model_temperatures = dict(model_temperatures or {})
        self._model_stop_sequences = {
            model: tuple(sequences) for model, sequences in (model_stop_sequences or {}).items()
        }

    def _url(self, model: str) -> str:
        base_url = self._model_base_urls.get(model)
        if base_url is None:
            raise ModelGatewayError(
                f"No endpoint is configured for model {model!r}", reason="no_endpoint"
            )
        if callable(base_url):
            base_url = base_url()
            if base_url is None:
                raise ModelGatewayError(
                    f"No running endpoint for model {model!r}", reason="unavailable"
                )
            base_url = base_url.rstrip("/")
        return f"{base_url}/chat/completions"

    def _payload(self, model: str, **fields: Any) -> dict[str, Any]:
        # 상한이 없으면 OpenRouter는 모델 최대 출력 토큰을 크레딧에서 예약해 402를 낸다.
        payload: dict[str, Any] = {"model": model, **fields}
        max_tokens = self._model_max_tokens.get(model)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if model in self._model_enable_thinking:
            payload["enable_thinking"] = self._model_enable_thinking[model]
        if model in self._model_temperatures:
            payload["temperature"] = self._model_temperatures[model]
        if model in self._model_stop_sequences:
            payload["stop"] = list(self._model_stop_sequences[model])
        return payload

    def _headers(self) -> dict[str, str]:
        if self._api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _post(self, model: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            response = await self._client.post(
                self._url(model),
                headers=self._headers(),
                json=payload,
                **self._request_options,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise request_error(error, "Model endpoint request failed") from error
        if not isinstance(body, Mapping):
            raise ModelGatewayError(
                "Model endpoint returned a non-object response", reason="invalid_json"
            )
        # OpenRouter는 공급자 오류를 200 응답 본문의 error로 돌려주기도 한다.
        provider, detail = _upstream_error(body)
        if detail is not None:
            raise ModelGatewayError(
                "Model endpoint returned an error body",
                reason="upstream_error",
                provider=provider,
                detail=detail,
            )
        return body

    @staticmethod
    def _content(body: Mapping[str, Any]) -> str:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelGatewayError(
                "Model endpoint response has no message content",
                reason="no_content",
                provider=str(body.get("provider") or "") or None,
            ) from error
        if not isinstance(content, str) or not content.strip():
            # 길이 제한(length)이나 안전 필터(content_filter)로 비어 오는 경우를 구분한다.
            raise ModelGatewayError(
                "Model endpoint returned empty message content",
                reason="empty_content",
                provider=str(body.get("provider") or "") or None,
                detail=f"finish_reason={choice.get('finish_reason')}",
            )
        return content.strip()

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        body = await self._post(
            model,
            self._payload(model, messages=list(messages), stream=False),
        )
        return self._content(body)

    async def structured(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        body = await self._post(
            model,
            self._payload(
                model,
                messages=list(messages),
                stream=False,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            ),
        )
        content = self._content(body)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise ModelGatewayError(
                "Structured model response is not valid JSON", reason="invalid_structured"
            ) from error
        if not isinstance(parsed, Mapping):
            raise ModelGatewayError(
                "Structured model response must be a JSON object", reason="invalid_structured"
            )
        return parsed

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
    ) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST",
                self._url(model),
                headers=self._headers(),
                json=self._payload(model, messages=list(messages), stream=True),
                **self._request_options,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    chunk = json.loads(payload)
                    text = chunk["choices"][0]["delta"].get("content")
                    if text:
                        yield text
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            raise request_error(error, "Model streaming request failed") from error
