import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from admin_api import credentials
from admin_api.client import completion, probe
from admin_api.credentials import connection_key
from admin_api.main import create_app
from admin_api.store import Store
from tests.unit.test_needu_lab import extraction, settings

CATALOGUE = {
    "data": [
        {
            "id": "Qwen/Qwen3.5-9B",
            "providers": [
                {
                    "provider": "deepinfra",
                    "status": "live",
                    "supports_structured_output": True,
                    "context_length": 8192,
                },
                {"provider": "unsupported", "status": "live", "supports_structured_output": False},
                {"provider": "pending", "status": "staging"},
            ],
        }
    ]
}


def test_public_catalogue_without_token_is_not_inference_success(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(credentials, "ROOT", tmp_path)
    store = Store(tmp_path / "lab.db")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=CATALOGUE)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe(client, store.connection("huggingface"))

    result = asyncio.run(exercise())
    assert not result["ok"]
    assert not result["credential_configured"]
    assert not result["inference_verified"]
    assert result["models"] == ["Qwen/Qwen3.5-9B:deepinfra", "Qwen/Qwen3.5-9B:unsupported"]
    assert "HF_TOKEN" in result["error"]
    assert "authorization" not in seen[0].headers
    assert result["model_metadata"][0]["supports_structured_output"]


def test_dotenv_token_resolution_and_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_synthetic_file_token\n")
    connection = {"api_key_env": "HF_TOKEN"}
    assert connection_key(connection) == "hf_synthetic_file_token"
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic_environment_token")
    assert connection_key(connection) == "hf_synthetic_environment_token"
    monkeypatch.delenv("HF_TOKEN")
    assert connection_key(connection, settings(hf_token="hf_settings_token")) == "hf_settings_token"
    with pytest.raises(ValueError, match="MISSING_LAB_KEY"):
        connection_key({"api_key_env": "MISSING_LAB_KEY"})


def test_hf_chat_uses_pinned_provider_token_and_real_extraction(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic")
    seen = []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer hf_synthetic"
        assert request.url.host == "router.huggingface.co"
        if request.method == "GET":
            return httpx.Response(200, json=CATALOGUE)
        data = json.loads(request.content)
        seen.append(data)
        assert data["model"] == "Qwen/Qwen3.5-9B:deepinfra"
        output = (
            json.dumps(extraction(data["messages"][-1]["content"]), ensure_ascii=False)
            if "response_format" in data
            else "캠핑할 때 어떤 순간이 좋아요?"
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    app = create_app(
        tmp_path / "lab.db",
        transport=httpx.MockTransport(handler),
        service_settings=settings(hf_token="hf_synthetic"),
    )
    with TestClient(app) as client:
        connections = client.get("/api/connections").json()
        assert any(c["provider"] == "huggingface" for c in connections)
        inspected = client.post("/api/connections/huggingface/probe").json()
        assert inspected["ok"]
        assert not inspected["inference_verified"]
        created = client.post(
            "/api/chats",
            json={
                "response_connection_id": "huggingface",
                "extraction_connection_id": "huggingface",
                "judgment_mode": "disabled",
            },
        )
        assert created.status_code == 201
        chat = created.json()
        assert chat["config"]["extraction"]["hosting"]["provider"] == "deepinfra"
        sent = client.post(
            f"/api/chats/{chat['id']}/messages",
            json={"userId": 1, "message": "주말마다 캠핑을 가요."},
        )
        assert sent.status_code == 200
        assert sent.json()["turn_count"] == 1
        assert not sent.json()["last_extraction_failed"]
        assert "hf_synthetic" not in json.dumps(sent.json())
        assert any("response_format" in payload for payload in seen)


def test_known_unsupported_hf_provider_rejected_for_extraction(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic")
    app = create_app(
        tmp_path / "lab.db",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=CATALOGUE)),
        service_settings=settings(),
    )
    with TestClient(app) as client:
        connection = next(
            c for c in client.get("/api/connections").json() if c["id"] == "huggingface"
        )
        connection["model"] = "Qwen/Qwen3.5-9B:unsupported"
        assert client.put("/api/connections/huggingface", json=connection).status_code == 200
        client.post("/api/connections/huggingface/probe")
        result = client.post(
            "/api/chats",
            json={
                "response_connection_id": "huggingface",
                "extraction_connection_id": "huggingface",
            },
        )
        assert result.status_code == 422
        assert "구조화 추출" in result.json()["detail"]


def test_preset_preserves_user_connection_on_restart(tmp_path):
    path = tmp_path / "lab.db"
    store = Store(path)
    connection = store.connection("huggingface")
    connection["model"] = "other/model:deepinfra"
    store.save_connection(
        "huggingface",
        {k: v for k, v in connection.items() if k not in ("id", "probe", "checked_at")},
    )
    assert Store(path).connection("huggingface")["model"] == "other/model:deepinfra"


@pytest.mark.parametrize("code,hint", [(401, "토큰"), (403, "권한"), (402, "크레딧")])
def test_hf_auth_billing_errors_are_sanitized(monkeypatch, code, hint):
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic")

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(code, text="private provider body hf_synthetic")
            )
        ) as client:
            return await completion(
                client,
                {
                    "model": "Qwen/model:provider",
                    "base_url": "https://router.huggingface.co/v1",
                    "api_key_env": "HF_TOKEN",
                },
                {"temperature": 0.2, "max_tokens": 30, "timeout_seconds": 2},
                [{"role": "user", "content": "hi"}],
            )

    result = asyncio.run(exercise())
    assert not result["ok"]
    assert hint in result["error"]
    assert "hf_synthetic" not in json.dumps(result)
    assert "private provider" not in json.dumps(result)


def test_model_name_resolves_live_route_and_rejects_unserved_models(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic")
    app = create_app(
        tmp_path / "lab.db",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=CATALOGUE)),
        service_settings=settings(),
    )
    with TestClient(app) as client:
        body = {
            "name": "HF candidate",
            "provider": "huggingface",
            "base_url": "https://router.huggingface.co/v1",
            "model": "Qwen/Qwen3.5-9B",
        }
        created = client.post("/api/connections", json=body)
        assert created.status_code == 201
        connection = created.json()
        assert connection["model"] == "Qwen/Qwen3.5-9B:deepinfra"
        assert connection["api_key_env"] == "HF_TOKEN"
        assert connection["environment"] == "other"
        assert connection["probe"]["selected_route"]["supports_structured_output"]
        assert not connection["probe"]["inference_verified"]
        bad = client.put(
            "/api/connections/" + connection["id"],
            json=body | {"model": "upstage/model-not-served"},
        )
        assert bad.status_code == 422
        assert "가중치" in bad.json()["detail"]
        saved = next(
            c for c in client.get("/api/connections").json() if c["id"] == connection["id"]
        )
        assert saved["model"] == connection["model"]
        assert (
            client.put(
                "/api/connections/" + connection["id"],
                json=body | {"model": "Qwen/Qwen3.5-9B:pending"},
            ).status_code
            == 422
        )
