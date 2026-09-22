import asyncio
import json

import httpx
import pytest

from app.infrastructure.embedding import EmbeddingError, OpenAICompatibleEmbedder


def _embedder(handler, *, batch_size: int = 100, dimensions: int = 3) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="https://api.upstage.ai/v1/",
        space_id="solar-embedding-2",
        passage_model="solar-embedding-2-passage",
        query_model="solar-embedding-2-query",
        dimensions=dimensions,
        api_key="test-key",
        batch_size=batch_size,
    )


def test_embeds_in_batches_and_restores_input_order() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        inputs = json.loads(request.content)["input"]
        rows = [
            {"index": index, "embedding": [float(len(text)), 0.0, 1.0]}
            for index, text in enumerate(inputs)
        ]
        return httpx.Response(200, json={"data": list(reversed(rows))})

    vectors = asyncio.run(_embedder(handler, batch_size=2).embed_passages(["a", "bb", "ccc"]))

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]
    assert len(requests) == 2
    assert str(requests[0].url) == "https://api.upstage.ai/v1/embeddings"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content)["model"] == "solar-embedding-2-passage"


def test_rejects_vectors_with_unexpected_dimensions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    with pytest.raises(EmbeddingError, match="3 dimensions"):
        asyncio.run(_embedder(handler).embed_passages(["티타늄 머그컵"]))


def test_rejects_empty_text_before_calling_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("endpoint must not be called")

    with pytest.raises(EmbeddingError, match="empty text"):
        asyncio.run(_embedder(handler).embed_queries(["캠핑", "  "]))


def test_wraps_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    with pytest.raises(EmbeddingError, match="request failed"):
        asyncio.run(_embedder(handler).embed_queries(["캠핑"]))


def test_omits_authorization_header_when_key_is_blank() -> None:
    headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0]}]})

    embedder = OpenAICompatibleEmbedder(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="https://api.upstage.ai/v1",
        space_id="solar-embedding-2",
        passage_model="solar-embedding-2-passage",
        query_model="solar-embedding-2-query",
        dimensions=3,
        api_key="",
    )
    asyncio.run(embedder.embed_queries(["캠핑"]))

    assert "Authorization" not in headers[0]


def test_uses_query_model_for_queries_and_passage_model_for_documents() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0]}]})

    embedder = _embedder(handler)
    asyncio.run(embedder.embed_queries(["캠핑장에서 핸드드립"]))
    asyncio.run(embedder.embed_passages(["티타늄 더블월 머그컵"]))

    assert models == ["solar-embedding-2-query", "solar-embedding-2-passage"]
    assert embedder.space_id == "solar-embedding-2"
