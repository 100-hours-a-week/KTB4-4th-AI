import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from admin_api.main import create_app
from admin_api.models import RunSpec
from admin_api.service_config import pipeline_configuration
from admin_api.store import Store
from benchmarks.needu import execute_needu
from benchmarks.persona_dialogue import PROMPT_VERSION, persona_system
from benchmarks.personas import persona_snapshot
from tests.unit.test_needu_lab import extraction, settings


def configured():
    return settings(extraction_model_name="google/gemini-test")


def enqueue(store, max_turns=3):
    spec = RunSpec(dialogue_mode="simulated", persona_id="p01", max_turns=max_turns)
    persona = persona_snapshot("p01")
    config = pipeline_configuration(configured(), spec, store)
    store.create_run(
        spec.model_dump(),
        {"model": "service"},
        {
            "pipeline": config,
            "persona": persona,
            "persona_simulator": {
                "prompt_version": PROMPT_VERSION,
                "system": persona_system(persona),
            },
        },
    )
    return store.claim()


def test_simulator_sees_actual_dialogue_without_review_answers(tmp_path):
    store = Store(tmp_path / "lab.db")
    run = enqueue(store)
    seen = []
    user_turns = ["주말마다 캠핑을 가요.", "캠핑 장비는 베이지색이 좋아요.", "강한 향은 싫어요."]

    def respond(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        schema = payload.get("response_format", {}).get("json_schema", {}).get("schema", {})
        if "message" in schema.get("properties", {}):
            index = (
                sum(
                    "message"
                    in p.get("response_format", {})
                    .get("json_schema", {})
                    .get("schema", {})
                    .get("properties", {})
                    for _, p in seen
                )
                - 1
            )
            output = json.dumps(
                {"message": user_turns[index], "end_conversation": index == 2}, ensure_ascii=False
            )
        elif schema:
            output = json.dumps(extraction(payload["messages"][-1]["content"]), ensure_ascii=False)
        else:
            output = f"니쥬 실제 응답 {len(seen)}: 어떤 순간이 좋았나요?"
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await execute_needu(store, run, settings=configured(), client=client)

    asyncio.run(exercise())
    result = store.run(run["id"])
    assert result["status"] == "completed"
    assert result["summary"]["executed_turns"] == 3
    assert result["summary"]["role_latency_p95_ms"]["persona"] is not None
    assert result["summary"]["role_latency_p95_ms"]["response"] is not None
    assert result["summary"]["stop_reason"] == "페르소나 사용자 종료"
    assert [s["input"] for s in result["samples"] if s["stage"] == "turn"] == user_turns
    persona_calls = [
        c for s in result["samples"] for c in s["model_calls"] if c["role"] == "persona"
    ]
    assert len(persona_calls) == 3
    assert persona_calls[0]["messages"][1]["role"] == "user"
    assert "니쥬 실제 응답 1" in persona_calls[0]["messages"][1]["content"]
    assert persona_calls[1]["messages"][2] == {"role": "assistant", "content": user_turns[0]}
    assert "니쥬 실제 응답" in persona_calls[1]["messages"][3]["content"]
    system = persona_calls[0]["messages"][0]["content"]
    for key in ("expected", "query_targets", "suggested_items", "situation", "avoid"):
        assert key not in system
    assert "휴대용 커피 도구" not in system
    assert "[STATE]" not in json.dumps(persona_calls, ensure_ascii=False)
    assert all(c["model"] == "google/gemini-test" for c in persona_calls)
    assert result["samples"][-1]["pipeline"]["queries"]
    review = result["samples"][-1]["pipeline"]["query_review"]
    assert result["summary"]["query_review"] == review
    assert 0 < review["score"] < 1
    assert review["exclusion_score"] == 1
    assert "expected_queries" not in system
    assert "forbidden_query_terms" not in system
    assert "synthetic-model-key" not in json.dumps(result)


@pytest.mark.parametrize(
    "bad_reply",
    [
        None,
        {"message": " ", "end_conversation": False},
        {"message": "가" * 501, "end_conversation": False},
    ],
)
def test_persona_failure_does_not_fall_back_to_script(tmp_path, bad_reply):
    store = Store(tmp_path / "lab.db")
    run = enqueue(store)

    def respond(request):
        payload = json.loads(request.content)
        if "response_format" in payload:
            if bad_reply is None:
                return httpx.Response(503, text="private provider detail")
            output = json.dumps(bad_reply)
        else:
            output = "안녕하세요."
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await execute_needu(store, run, settings=configured(), client=client)

    asyncio.run(exercise())
    result = store.run(run["id"])
    assert result["status"] == "failed"
    assert result["samples"][-1]["input"] == ""
    assert "private provider detail" not in json.dumps(result)
    assert result["samples"][-1]["model_calls"][0]["role"] == "persona"


def test_simulated_mode_validation_and_manifest(tmp_path):
    with pytest.raises(ValidationError):
        RunSpec(dialogue_mode="simulated")
    with pytest.raises(ValidationError):
        RunSpec(mode="load", prompt="test", dialogue_mode="simulated")
    with pytest.raises(ValidationError):
        RunSpec(dialogue_mode="simulated", persona_id="p01", max_turns=21)
    path = tmp_path / "lab.db"
    app = create_app(path, service_settings=configured())
    with TestClient(app) as client:
        app.state.store.beat()
        result = client.post(
            "/api/runs", json={"dialogue_mode": "simulated", "persona_id": "p01", "max_turns": 3}
        )
        assert result.status_code == 201
        run = result.json()
        assert run["spec"]["requests"] == 5
        assert run["manifest"]["persona_simulator"]["prompt_version"] == PROMPT_VERSION
        assert run["manifest"]["pipeline"]["persona"]["model"] == "google/gemini-test"
        assert (
            client.post(
                "/api/runs", json={"dialogue_mode": "simulated", "persona_id": "missing"}
            ).status_code
            == 422
        )
    with pytest.raises(ValueError, match="Gemini"):
        pipeline_configuration(
            settings(), RunSpec(dialogue_mode="simulated", persona_id="p01"), Store(path)
        )


def test_cancel_in_flight_persona_call(tmp_path):
    store = Store(tmp_path / "lab.db")
    run = enqueue(store)
    started = asyncio.Event()
    cancelled = []

    async def respond(request):
        if "response_format" in json.loads(request.content):
            started.set()
            try:
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        return httpx.Response(200, json={"choices": [{"message": {"content": "안녕하세요."}}]})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            task = asyncio.create_task(
                execute_needu(store, run, settings=configured(), client=client)
            )
            await asyncio.wait_for(started.wait(), 2)
            store.cancel(run["id"])
            await asyncio.wait_for(task, 2)

    asyncio.run(exercise())
    assert cancelled
    assert store.run(run["id"])["status"] == "cancelled"


def test_persona_api_stays_pinned_when_service_model_defaults_change(tmp_path, monkeypatch):
    from admin_api.service_config import model_key, persona_configuration

    monkeypatch.setenv("PERSONA_TEST_API_KEY", "synthetic-persona-key")
    dedicated = dict(
        persona_model_base_url="https://persona.example/v1",
        persona_model_name="google/gemini-test",
        persona_model_api_key_env="PERSONA_TEST_API_KEY",
    )
    api_settings = settings(**dedicated, model_api_key="synthetic-service-key")
    local_settings = settings(
        **dedicated,
        response_model_base_url="https://colab.example/v1",
        response_model_name="colab-response",
        extraction_model_base_url="https://colab.example/v1",
        extraction_model_name="colab-extract",
        model_api_key="synthetic-colab-key",
    )
    assert persona_configuration(api_settings) == persona_configuration(local_settings)
    assert (
        model_key(persona_configuration(local_settings), local_settings) == "synthetic-persona-key"
    )
    store = Store(tmp_path / "lab.db")
    spec = RunSpec(dialogue_mode="simulated", persona_id="p01", max_turns=1)
    config = pipeline_configuration(local_settings, spec, store)
    assert config["persona"]["base_url"] == "https://persona.example/v1"
    assert config["persona"]["model"] == "google/gemini-test"
    assert "synthetic-persona-key" not in json.dumps(config)


def test_persona_and_colab_calls_use_independent_endpoints_and_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSONA_TEST_API_KEY", "synthetic-persona-key")
    monkeypatch.setenv("AI_LAB_COLAB_KEY", "synthetic-colab-key")
    store = Store(tmp_path / "lab.db")
    connection = dict(
        name="Colab",
        provider="openai",
        environment="colab",
        base_url="https://colab.example/v1",
        model="loaded-colab-model",
        revision="",
        precision="BF16",
        hardware="A100",
        api_key_env="AI_LAB_COLAB_KEY",
    )
    store.save_connection("colab", connection)
    configured = settings(
        model_api_key="unrelated-service-key",
        persona_model_base_url="https://persona.example/v1",
        persona_model_name="google/gemini-test",
        persona_model_api_key_env="PERSONA_TEST_API_KEY",
    )
    spec = RunSpec(
        dialogue_mode="simulated",
        persona_id="p01",
        max_turns=1,
        response_connection_id="colab",
        extraction_connection_id="colab",
        judgment_mode="disabled",
    )
    persona = persona_snapshot("p01")
    config = pipeline_configuration(configured, spec, store)
    store.create_run(
        spec.model_dump(),
        {"model": "service"},
        {
            "pipeline": config,
            "persona": persona,
            "persona_simulator": {
                "prompt_version": PROMPT_VERSION,
                "system": persona_system(persona),
            },
        },
    )
    run = store.claim()
    # A new loaded model changes the next experiment, without touching the persona API.
    store.save_connection("colab", {**connection, "model": "next-colab-model"})
    assert pipeline_configuration(configured, spec, store)["persona"] == config["persona"]
    seen = []

    def respond(request):
        payload = json.loads(request.content)
        seen.append(request.url.host)
        if request.url.host == "persona.example":
            assert request.headers["Authorization"] == "Bearer synthetic-persona-key"
            assert payload["model"] == "google/gemini-test"
            output = json.dumps({"message": "주말마다 캠핑을 가요.", "end_conversation": True})
        else:
            assert request.url.host == "colab.example"
            assert request.headers["Authorization"] == "Bearer synthetic-colab-key"
            assert payload["model"] == "loaded-colab-model"
            output = (
                json.dumps(extraction(payload["messages"][-1]["content"]))
                if "response_format" in payload
                else "캠핑할 때 어떤 순간이 좋아요?"
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await execute_needu(store, run, settings=configured, client=client)

    asyncio.run(exercise())
    saved = store.run(run["id"])
    assert saved["status"] == "completed"
    assert {"persona.example", "colab.example"} <= set(seen)
    assert "synthetic-persona-key" not in json.dumps(saved)
    assert "synthetic-colab-key" not in json.dumps(saved)


def test_missing_dedicated_persona_key_does_not_fall_back_to_colab_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PERSONA_MISSING_KEY", raising=False)
    configured = settings(
        model_api_key="synthetic-colab-key",
        persona_model_base_url="https://persona.example/v1",
        persona_model_name="google/gemini-test",
        persona_model_api_key_env="PERSONA_MISSING_KEY",
    )
    with pytest.raises(ValueError, match="PERSONA_MISSING_KEY"):
        pipeline_configuration(
            configured,
            RunSpec(dialogue_mode="simulated", persona_id="p01"),
            Store(tmp_path / "lab.db"),
        )
