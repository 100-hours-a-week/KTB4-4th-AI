from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx


class EmbeddingError(RuntimeError):
    """Raised when an embedding endpoint cannot produce valid vectors."""


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        space_id: str,
        passage_model: str,
        query_model: str,
        dimensions: int,
        api_key: str | None = None,
        batch_size: int = 100,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = client
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._space_id = space_id
        self._passage_model = passage_model
        self._query_model = query_model
        self._dimensions = dimensions
        self._api_key = api_key or None
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds

    @property
    def space_id(self) -> str:
        return self._space_id

    def _headers(self) -> dict[str, str]:
        if self._api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _post(self, model: str, batch: Sequence[str]) -> Mapping[str, Any]:
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers(),
                json={"model": model, "input": list(batch)},
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise EmbeddingError("Embedding endpoint request failed") from error
        if not isinstance(body, Mapping):
            raise EmbeddingError("Embedding endpoint returned a non-object response")
        return body

    def _vectors(self, body: Mapping[str, Any], expected: int) -> list[list[float]]:
        try:
            rows = sorted(body["data"], key=lambda row: row["index"])
            vectors = [[float(value) for value in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError("Embedding endpoint response has no vectors") from error
        if len(vectors) != expected:
            raise EmbeddingError(f"Expected {expected} vectors, received {len(vectors)}")
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"Expected {self._dimensions} dimensions, received {len(vector)}"
                )
        return vectors

    async def _embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Embedding input must not contain empty text")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._vectors(await self._post(model, batch), len(batch)))
        return vectors

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(self._passage_model, texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(self._query_model, texts)
