from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


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
