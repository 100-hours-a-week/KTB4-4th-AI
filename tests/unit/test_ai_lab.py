import asyncio
import json

import httpx

from admin_api.client import completion
from admin_api.main import create_app
from admin_api.models import Connection, RunSpec
from admin_api.store import Store
from benchmarks.fixtures import CASES, evaluate
from benchmarks.runner import execute, summarize


def stream_response(content="안녕하세요", usage=None, terminated=True):
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "internal"}}]},
        {
            "choices": [
                {"delta": {"content": content}, "finish_reason": "stop" if terminated else None}
            ]
        },
    ]
    if usage is not None:
        chunks.append({"choices": [], "usage": usage})
    text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    if terminated:
        text += "data: [DONE]\n\n"
    return httpx.Response(200, text=text, headers={"Content-Type": "text/event-stream"})


def test_stream_preserves_usage_and_detects_interruption():
    async def exercise():
        connection = {"model": "test-model", "base_url": "http://local/v1"}
        spec = {"temperature": 0.2, "max_tokens": 50, "timeout_seconds": 3}
        for terminated in (True, False):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _, ended=terminated: stream_response(
                        usage={"completion_tokens": 7}, terminated=ended
                    )
                )
            ) as client:
                result = await completion(
                    client, connection, spec, [{"role": "user", "content": "hi"}]
                )
                assert result["ok"] is terminated
                assert result["output"] == "안녕하세요"
                assert result["usage"]["completion_tokens"] == 7
                assert result["first_chunk_ms"] >= 0

    asyncio.run(exercise())


def test_missing_usage_is_not_an_estimated_token_count():
    result = summarize(
        [
            {"ok": True, "latency_ms": 10, "first_chunk_ms": 2, "usage": {"completion_tokens": 7}},
            {"ok": True, "latency_ms": 30, "first_chunk_ms": 4, "usage": None},
            {"ok": False, "latency_ms": 100, "expected": {}, "evaluation": {"schema_valid": False}},
        ],
        elapsed=2,
        planned=3,
    )
    assert result["success_rate"] == 2 / 3
    assert result["latency_p50_ms"] == 20
    assert result["latency_p95_ms"] == 29
    assert result["output_tokens_per_second"] is None
    assert result["exact_match_rate"] == 0
    assert result["complete"]


def test_quality_schema_and_content_are_independent():
    expected = CASES[0]["expected"]
    assert evaluate(json.dumps(expected), expected)["exact_match"]
    wrong = {"likes": [], "dislikes": ["커피"], "activities": []}
    result = evaluate(json.dumps(wrong), expected)
    assert result["schema_valid"]
    assert not result["exact_match"]
    assert not evaluate('{"likes": "커피"}', expected)["schema_valid"]
    duplicate = {**expected, "likes": ["커피", "커피"]}
    assert not evaluate(json.dumps(duplicate), expected)["exact_match"]


def test_api_snapshot_queue_cancel_delete_and_origin(tmp_path):
    app = create_app(tmp_path / "lab.sqlite3")

    async def exercise():
        async with app.router.lifespan_context(app):
            app.state.store.beat()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                updated = await client.put(
                    "/api/connections/lmstudio",
                    json={
                        "name": "Local",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "model": "model-v1",
                    },
                )
                assert updated.status_code == 200
                response = await client.post(
                    "/api/runs",
                    json={
                        "connection_id": "lmstudio",
                        "prompt": "안녕",
                        "mode": "load",
                        "requests": 1,
                    },
                )
                assert response.status_code == 201
                run = response.json()
                assert run["spec"]["requests"] == 1
                assert run["manifest"]["fixture_hash"]
                await client.put(
                    "/api/connections/lmstudio",
                    json={
                        "name": "Changed",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "model": "model-v2",
                    },
                )
                detail = (await client.get(f"/api/runs/{run['id']}")).json()
                assert detail["connection"]["model"] == "model-v1"
                assert (await client.post(f"/api/runs/{run['id']}/cancel")).status_code == 200
                assert app.state.store.run(run["id"])["status"] == "cancelled"
                assert (await client.delete(f"/api/runs/{run['id']}")).status_code == 200
                assert app.state.store.run(run["id"]) is None
                rejected = await client.post(
                    "/api/connections",
                    json={"name": "Bad", "base_url": "http://127.0.0.1:1234/v1"},
                    headers={"Origin": "https://unrelated.example"},
                )
                assert rejected.status_code == 403
                capped = await client.post(
                    "/api/runs",
                    json={
                        "connection_id": "lmstudio",
                        "mode": "load",
                        "prompt": "hi",
                        "requests": 101,
                    },
                )
                assert capped.status_code == 422
                removed = await client.post(
                    "/api/runs", json={"mode": "playground", "prompt": "안녕"}
                )
                assert removed.status_code == 422

    asyncio.run(exercise())


def test_worker_runs_actual_fixture_requests_with_schema_and_persists_failures(tmp_path):
    store = Store(tmp_path / "lab.sqlite3")
    spec = RunSpec(connection_id="lmstudio", mode="quality", concurrency=2).model_dump()
    spec["requests"] = len(CASES)
    connection = Connection(name="Mock", base_url="http://local/v1", model="model").model_dump()
    store.create_run(spec, connection, {})
    run = store.claim()
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        prompt = payload["messages"][-1]["content"]
        case = next(c for c in CASES if c["input"] == prompt)
        if case["id"] == "negative":
            return httpx.Response(503, text="Unavailable")
        return stream_response(json.dumps(case["expected"]), {"completion_tokens": 20})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await execute(store, run, client=client)

    asyncio.run(exercise())
    saved = store.run(run["id"])
    assert saved["status"] == "completed"
    assert len(saved["samples"]) == 8
    assert saved["summary"]["exact_match_rate"] == 7 / 8
    assert saved["summary"]["failed"] == 1
    assert all(payload["response_format"]["json_schema"]["strict"] for payload in seen)
    assert store.delete(run["id"])
    with store.db() as db:
        assert db.execute("SELECT count(*) FROM samples").fetchone()[0] == 0


def test_cancel_stops_inflight_requests_and_does_not_issue_queued_requests(tmp_path):
    store = Store(tmp_path / "lab.sqlite3")
    spec = RunSpec(connection_id="lmstudio", mode="load", prompt="hello", requests=10).model_dump()
    store.create_run(spec, {"model": "test", "base_url": "http://local/v1"}, {})
    run = store.claim()
    started = asyncio.Event()
    cancelled = []

    async def handler(_):
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return stream_response()

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(execute(store, run, client=client))
            await started.wait()
            store.cancel(run["id"])
            await asyncio.wait_for(task, 2)

    asyncio.run(exercise())
    saved = store.run(run["id"])
    assert saved["status"] == "cancelled"
    assert len(saved["samples"]) == 1
    assert saved["summary"]["failed"] == 1
    assert not saved["summary"]["complete"]
    assert cancelled


def test_recovery_marks_interrupted_without_requeue(tmp_path):
    store = Store(tmp_path / "lab.sqlite3")
    spec = RunSpec(connection_id="lmstudio", mode="load", prompt="hello").model_dump()
    created = store.create_run(spec, {"model": "test"}, {})
    assert store.claim()["id"] == created["id"]
    store.recover()
    assert store.run(created["id"])["status"] == "interrupted"
    assert store.claim() is None


def test_queued_quality_run_uses_recorded_fixture_instead_of_current_cases(tmp_path):
    store = Store(tmp_path / "lab.sqlite3")
    spec = RunSpec(connection_id="lmstudio", mode="quality").model_dump()
    case = {"label": "스냅샷 사례", "input": "frozen", "expected": CASES[0]["expected"]}
    store.create_run(
        spec,
        {"model": "test", "base_url": "http://local/v1"},
        {"fixture": {"cases": [case], "system": "frozen system", "schema": {"type": "object"}}},
    )
    run = store.claim()

    def handler(request):
        payload = json.loads(request.content)
        assert payload["messages"][0]["content"] == "frozen system"
        assert payload["messages"][1]["content"] == "frozen"
        return stream_response(json.dumps(case["expected"]))

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await execute(store, run, client=client)

    asyncio.run(exercise())
    assert len(store.run(run["id"])["samples"]) == 1
    assert store.run(run["id"])["summary"]["exact_match_rate"] == 1


def test_exports_escape_csv_formulas_and_preserve_full_json(tmp_path):
    app = create_app(tmp_path / "lab.sqlite3")

    async def exercise():
        async with app.router.lifespan_context(app):
            store = app.state.store
            run = store.create_run({"mode": "playground"}, {"model": "test"}, {})
            store.sample(
                run["id"],
                0,
                {
                    "ok": False,
                    "label": "=SUM(1,1)",
                    "input": "synthetic input",
                    "output": "synthetic output",
                    "error": "@provider error",
                },
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                csv_result = await client.get(f"/api/runs/{run['id']}/export?format=csv")
                assert csv_result.status_code == 200
                assert "'=SUM(1,1)" in csv_result.text
                assert "'@provider error" in csv_result.text
                assert "synthetic input" not in csv_result.text
                full = await client.get(f"/api/runs/{run['id']}/export?format=json")
                assert full.json()["samples"][0]["output"] == "synthetic output"
                assert (
                    await client.get(f"/api/runs/{run['id']}/export?format=html")
                ).status_code == 422

    asyncio.run(exercise())


def test_key_is_loaded_from_worker_environment_and_not_returned(monkeypatch):
    monkeypatch.setenv("AI_LAB_TEST_KEY", "test-only-secret")

    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-only-secret"
        return stream_response()

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await completion(
                client,
                {"model": "test", "base_url": "http://local/v1", "api_key_env": "AI_LAB_TEST_KEY"},
                {"max_tokens": 10, "temperature": 0, "timeout_seconds": 2},
                [{"role": "user", "content": "synthetic"}],
            )
        assert result["ok"]
        assert "test-only-secret" not in json.dumps(result)

    asyncio.run(exercise())
