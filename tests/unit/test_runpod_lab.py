import pytest
from fastapi.testclient import TestClient

from admin_api import credentials
from admin_api.credentials import connection_key
from admin_api.main import create_app
from tests.unit.test_needu_lab import settings


def test_runpod_key_from_dotenv_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "ROOT", tmp_path)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    (tmp_path / ".env").write_text("MODEL_API_KEY=synthetic-pod-key\n")
    connection = {"api_key_env": "MODEL_API_KEY"}
    assert connection_key(connection) == "synthetic-pod-key"
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-override")
    assert connection_key(connection) == "synthetic-override"
    monkeypatch.delenv("MODEL_API_KEY")
    (tmp_path / ".env").write_text("MODEL_API_KEY=\n")
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        connection_key(connection)


def test_runpod_import_follows_service_urls_without_storing_secrets(tmp_path):
    configured = settings(
        response_model_base_url="https://new-pod-8000.proxy.runpod.net/v1",
        extraction_model_base_url="https://new-pod-8000.proxy.runpod.net/v1",
        response_model_name="kanana-response",
        extraction_model_name="kanana-extractor",
    )
    app = create_app(tmp_path / "lab.db", service_settings=configured)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post("/api/connections/runpod")
            assert response.status_code == 200
            rows = response.json()
            assert [row["id"] for row in rows] == ["runpod-response", "runpod-extraction"]
            assert all(row["environment"] == "runpod" for row in rows)
            assert all(row["api_key_env"] == "MODEL_API_KEY" for row in rows)
            assert rows[1]["model"] == "kanana-extractor"
            assert "synthetic-model-key" not in response.text
        assert len([c for c in app.state.store.connections() if c["environment"] == "runpod"]) == 2
        configured.response_model_base_url = "https://moved-pod-8000.proxy.runpod.net/v1"
        assert client.post("/api/connections/runpod").json()[0]["base_url"] == str(
            configured.response_model_base_url
        )


def test_runpod_import_validates_both_roles_before_saving(tmp_path):
    configured = settings(response_model_base_url="https://pod-8000.proxy.runpod.net/v1")
    app = create_app(tmp_path / "lab.db", service_settings=configured)
    with TestClient(app) as client:
        assert client.post("/api/connections/runpod").status_code == 422
        assert app.state.store.connection("runpod-response") is None
