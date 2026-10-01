from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from app.infrastructure.model_gateway.openai_compatible import ModelGatewayError


class SystemOneJudgmentGateway:
    """TypeSafe System One 판단 API. OpenRouter는 같은 형식을 /systemone 으로 중계한다."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str | None = None,
    ) -> None:
        self._client = client
        self._url = f"{base_url.rstrip('/')}/systemone"
        self._api_key = api_key

    async def decide(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        model: str,
    ) -> Mapping[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = await self._client.post(
                self._url,
                headers=headers,
                json={"model": model, "state": state, "questions": questions},
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ModelGatewayError("Judgment endpoint request failed") from error
        if not isinstance(body, Mapping) or not isinstance(body.get("answers"), Mapping):
            raise ModelGatewayError("Judgment endpoint response has no answers")
        return body
