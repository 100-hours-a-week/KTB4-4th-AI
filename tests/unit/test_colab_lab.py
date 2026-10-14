import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from admin_api import credentials
from admin_api.client import completion
from admin_api.credentials import connection_key
from admin_api.main import create_app
from admin_api.models import RunSpec
from admin_api.store import ROOT
from notebooks.colab_transformers_api import create_model_api
from tests.unit.test_ai_lab import stream_response
from tests.unit.test_needu_lab import extraction, settings

COLAB_CONNECTION = {
    "name": "Colab · Qwen3-8B",
    "provider": "openai",
    "environment": "colab",
    "base_url": "https://colab-test.ngrok-free.app/v1",
    "model": "Qwen/Qwen3-8B",
    "api_key_env": "AI_LAB_COLAB_KEY",
    "revision": "test-model-commit",
    "precision": "BF16",
    "hardware": "A100 40GB · Transformers",
}


def test_colab_key_from_dotenv_environment_precedence_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "ROOT", tmp_path)
    monkeypatch.delenv("AI_LAB_COLAB_KEY", raising=False)
    (tmp_path / ".env").write_text("AI_LAB_COLAB_KEY=synthetic-file-key\n")
    assert connection_key(COLAB_CONNECTION) == "synthetic-file-key"
    monkeypatch.setenv("AI_LAB_COLAB_KEY", "synthetic-env-key")
    assert connection_key(COLAB_CONNECTION) == "synthetic-env-key"
    monkeypatch.delenv("AI_LAB_COLAB_KEY")
    (tmp_path / ".env").write_text("AI_LAB_COLAB_KEY=\n")
    with pytest.raises(ValueError, match="AI_LAB_COLAB_KEY"):
        connection_key(COLAB_CONNECTION)


def test_colab_connection_probe_chat_extraction_and_reconnect(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_LAB_COLAB_KEY", "synthetic-colab-key")
    seen = []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer synthetic-colab-key"
        assert request.url.host == "colab-test.ngrok-free.app"
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": COLAB_CONNECTION["model"]}]})
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        seen.append(payload)
        assert payload["model"] == COLAB_CONNECTION["model"]
        output = (
            json.dumps(extraction(payload["messages"][-1]["content"]), ensure_ascii=False)
            if "response_format" in payload
            else "캠핑하면서 어떤 순간이 좋아요?"
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    app = create_app(
        tmp_path / "lab.db",
        transport=httpx.MockTransport(handler),
        service_settings=settings(),
    )
    with TestClient(app) as client:
        registered = client.post("/api/connections", json=COLAB_CONNECTION)
        assert registered.status_code == 201
        connection = registered.json()
        connection_id = connection["id"]
        assert connection["environment"] == "colab"
        checked = client.post(f"/api/connections/{connection_id}/probe").json()
        assert checked["ok"]
        assert checked["models"] == [COLAB_CONNECTION["model"]]
        started = client.post(
            "/api/chats",
            json={
                "response_connection_id": connection_id,
                "extraction_connection_id": connection_id,
                "judgment_mode": "disabled",
            },
        )
        assert started.status_code == 201
        session_id = started.json()["id"]
        sent = client.post(
            f"/api/chats/{session_id}/messages",
            json={"userId": 1, "message": "주말마다 캠핑을 가요."},
        )
        assert sent.status_code == 200
        assert sent.json()["turn_count"] == 1
        assert not sent.json()["last_extraction_failed"]
        assert any(p.get("response_format", {}).get("type") == "json_schema" for p in seen)
        assert "synthetic-colab-key" not in sent.text
        assert "synthetic-colab-key" not in client.get("/api/overview").text
        app.state.store.beat()
        run = client.post(
            "/api/runs", json={"connection_id": connection_id, "mode": "load", "prompt": "안녕"}
        )
        assert run.status_code == 201
        assert run.json()["connection"]["environment"] == "colab"
        updated = client.put(
            f"/api/connections/{connection_id}",
            json={**COLAB_CONNECTION, "base_url": "https://restarted.ngrok-free.app/v1"},
        )
        assert updated.status_code == 200
        assert updated.json()["probe"] is None
        frozen = client.get(f"/api/runs/{run.json()['id']}").json()
        assert frozen["connection"]["base_url"] == COLAB_CONNECTION["base_url"]


def test_colab_streaming_load_request_keeps_usage_and_authentication(monkeypatch):
    monkeypatch.setenv("AI_LAB_COLAB_KEY", "synthetic-stream-key")

    def handler(request):
        assert request.headers["Authorization"] == "Bearer synthetic-stream-key"
        payload = json.loads(request.content)
        assert payload["stream"]
        assert payload["stream_options"]["include_usage"]
        return stream_response(usage={"completion_tokens": 12})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await completion(
                client,
                COLAB_CONNECTION,
                RunSpec(mode="load", prompt="안녕").model_dump(),
                [{"role": "user", "content": "안녕"}],
            )

    result = asyncio.run(exercise())
    assert result["ok"]
    assert result["usage"]["completion_tokens"] == 12


def test_downloaded_colab_notebook_is_valid_unexecuted_python(tmp_path):
    with TestClient(create_app(tmp_path / "lab.db")) as client:
        response = client.get("/api/colab/notebook")
    assert response.status_code == 200
    assert "ai_lab_colab.ipynb" in response.headers["content-disposition"]
    notebook = response.json()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert not cell["outputs"]
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), f"colab-cell-{index}", "exec")


def model_api(generator):
    return TestClient(
        create_model_api(["test-model", "second-model"], "synthetic-private-key", generator)
    )


def fake_generation(body, messages, started):
    yield "안녕 "
    yield "반가워요"
    yield {
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        "finish_reason": "stop",
        "timings": {"server_first_token_ms": 5},
    }


AUTH = {"Authorization": "Bearer synthetic-private-key"}


def test_transformers_api_auth_models_and_normal_completion():
    with model_api(fake_generation) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )
        response = client.get("/v1/models", headers=AUTH)
        assert [row["id"] for row in response.json()["data"]] == ["test-model", "second-model"]
        assert client.get("/docs", headers=AUTH).status_code == 404
        completed = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "안녕"}],
            },
        )
        assert completed.status_code == 200
        assert completed.json()["choices"][0]["message"]["content"] == "안녕 반가워요"
        assert completed.json()["usage"]["completion_tokens"] == 3
        assert completed.json()["timings"]["server_first_token_ms"] == 5
        assert "synthetic-private-key" not in completed.text
        unknown = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "unknown",
                "messages": [{"role": "user", "content": "안녕"}],
            },
        )
        assert unknown.status_code == 404


@pytest.mark.parametrize("stream", [False, True])
def test_transformers_schema_validation_reports_invalid_generation(stream):
    with model_api(fake_generation) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "추출"}],
                "stream": stream,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test",
                        "strict": True,
                        "schema": {"type": "object"},
                    },
                },
            },
        )
        if stream:
            assert '"error"' in response.text
            assert "[DONE]" not in response.text
            assert '"finish_reason": "stop"' not in response.text
        else:
            assert response.status_code == 422


def test_transformers_schema_is_added_to_prompt_and_checked_after_generation():
    schema = {
        "type": "object",
        "properties": {"likes": {"type": "array", "items": {"type": "string"}}},
        "required": ["likes"],
        "additionalProperties": False,
    }

    def generate(body, messages, started):
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("original instruction")
        assert json.dumps(schema, ensure_ascii=False) in messages[0]["content"]
        yield '{"likes": ["커피"]}'
        yield {"usage": {"prompt_tokens": 5, "completion_tokens": 8}, "finish_reason": "stop"}

    with model_api(generate) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "original instruction"},
                    {"role": "user", "content": "커피 좋아해"},
                ],
                "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
            },
        )
        assert response.status_code == 200
        assert json.loads(response.json()["choices"][0]["message"]["content"])["likes"] == ["커피"]


def test_transformers_stream_is_compatible_with_existing_lab_client():
    api = create_model_api(["test-model"], "synthetic-private-key", fake_generation)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api)) as client:
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("AI_LAB_COLAB_KEY", "synthetic-private-key")
                return await completion(
                    client,
                    {
                        "base_url": "http://testserver/v1",
                        "model": "test-model",
                        "api_key_env": "AI_LAB_COLAB_KEY",
                    },
                    RunSpec(mode="load", prompt="안녕").model_dump(),
                    [{"role": "user", "content": "안녕"}],
                )

    result = asyncio.run(exercise())
    assert result["ok"]
    assert result["output"] == "안녕 반가워요"
    assert result["usage"]["completion_tokens"] == 3


def test_notebook_embeds_api_source_without_serving_dependencies():
    notebook = json.loads((ROOT / "notebooks/ai_lab_colab.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert (ROOT / "notebooks/colab_transformers_api.py").read_text() in source
    assert "pip', 'install', '-q'" in source
    assert "'vllm'" not in source
    assert "VLLM_VERSION" not in source
    assert "transformers_generator(loaded" in source


@pytest.mark.parametrize("failed", [False, True])
def test_transformers_generator_streams_real_callback_and_releases_lock(monkeypatch, failed):
    import queue
    import sys
    import threading
    import time
    from contextlib import nullcontext
    from types import SimpleNamespace

    from notebooks.colab_transformers_api import CompletionRequest, transformers_generator

    class Streamer:
        def __init__(self, tokenizer, **kwargs):
            self.next_tokens_are_prompt = True
            self.events = queue.Queue()

        def put(self, value):
            if self.next_tokens_are_prompt:
                self.next_tokens_are_prompt = False
            else:
                self.events.put(value)

        def end(self):
            self.events.put(None)

        def __iter__(self):
            return self

        def __next__(self):
            text = self.events.get(timeout=1)
            if text is None:
                raise StopIteration
            return text

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            TextIteratorStreamer=Streamer,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        ),
    )

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert not kwargs["enable_thinking"]
            return "formatted prompt"

        def __call__(self, prompt, **kwargs):
            return Batch(input_ids=SimpleNamespace(shape=(1, 3)))

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert not kwargs["do_sample"]
            assert kwargs["pad_token_id"] == 0
            assert kwargs["max_new_tokens"] == 5
            if failed:
                raise RuntimeError("synthetic generation failure")
            kwargs["streamer"].put("prompt")
            kwargs["streamer"].put("안녕 ")
            kwargs["streamer"].put("반가워요")
            kwargs["streamer"].end()
            return SimpleNamespace(shape=(1, 5))

    lock = threading.Lock()
    generate = transformers_generator(
        {
            "test-model": {
                "model": Model(),
                "tokenizer": Tokenizer(),
                "lock": lock,
            }
        },
        max_context_tokens=16,
        timeout_seconds=2,
    )
    body = CompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "안녕"}],
        temperature=0,
        max_tokens=5,
    )
    if failed:
        with pytest.raises(Exception, match="모델 생성 실패"):
            list(generate(body, [], time.perf_counter()))
    else:
        events = list(generate(body, [], time.perf_counter()))
        assert "".join(events[:-1]) == "안녕 반가워요"
        assert events[-1]["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert events[-1]["timings"]["server_first_token_ms"] >= 0
    assert not lock.locked()
