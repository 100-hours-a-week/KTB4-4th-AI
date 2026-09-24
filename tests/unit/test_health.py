from collections.abc import Sequence

import asyncpg
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class FakeCatalogPool:
    async def close(self) -> None:
        return None


class FakeEmbedder:
    @property
    def space_id(self) -> str:
        return "test-embedding"

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_health_does_not_require_service_token() -> None:
    app = create_app(Settings(app_env="test", app_version="test-version"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "test-version"}
    assert response.headers["X-Request-ID"].startswith("req_")


def test_health_preserves_request_id() -> None:
    app = create_app(Settings(app_env="test"))

    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "req_from_backend"})

    assert response.headers["X-Request-ID"] == "req_from_backend"


def test_api_catalog_prefers_read_only_database_url(monkeypatch) -> None:
    connected_dsns: list[str] = []

    async def create_pool(*, dsn: str, min_size: int, max_size: int) -> FakeCatalogPool:
        connected_dsns.append(dsn)
        return FakeCatalogPool()

    monkeypatch.setattr(asyncpg, "create_pool", create_pool)
    app = create_app(
        Settings(
            app_env="test",
            catalog_database_url="postgresql://batch@localhost/catalog",
            catalog_database_url_ro="postgresql://api@localhost/catalog",
        ),
        embedder=FakeEmbedder(),
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert connected_dsns == ["postgresql://api@localhost/catalog"]
