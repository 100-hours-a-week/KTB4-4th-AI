from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from app.application.ports.model_gateway import Message


class ModelGatewayError(RuntimeError):
    """Raised when an OpenAI-compatible model endpoint cannot produce a valid response."""


class OpenAICompatibleModelGateway:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model_base_urls: Mapping[str, str],
        api_key: str | None = None,
        model_max_tokens: Mapping[str, int] | None = None,
        model_enable_thinking: Mapping[str, bool] | None = None,
        model_temperatures: Mapping[str, float] | None = None,
        model_stop_sequences: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._client = client
        self._model_base_urls = {
            model: base_url.rstrip("/") for model, base_url in model_base_urls.items()
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
            raise ModelGatewayError(f"No endpoint is configured for model {model!r}")
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
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ModelGatewayError("Model endpoint request failed") from error
        if not isinstance(body, Mapping):
            raise ModelGatewayError("Model endpoint returned a non-object response")
        return body

    @staticmethod
    def _content(body: Mapping[str, Any]) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelGatewayError("Model endpoint response has no message content") from error
        if not isinstance(content, str) or not content.strip():
            raise ModelGatewayError("Model endpoint returned empty message content")
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
            raise ModelGatewayError("Structured model response is not valid JSON") from error
        if not isinstance(parsed, Mapping):
            raise ModelGatewayError("Structured model response must be a JSON object")
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
            raise ModelGatewayError("Model streaming request failed") from error
