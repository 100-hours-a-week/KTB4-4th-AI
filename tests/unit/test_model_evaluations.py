import asyncio

import httpx
import pytest

from admin_api.main import create_app
from admin_api.model_evaluations import model_evaluations, service_evaluations
from app.application.conversation_prompts import FRIEND_SUMMARY_SYSTEM


def config(model, precision="BF16"):
    return {"model": model, "base_url": "http://local/v1", "precision": precision}


def run(run_id, response="chat", extraction="extract", status="completed"):
    return {
        "id": run_id,
        "status": status,
        "created_at": "2026-10-14T00:00:00Z",
        "spec": {"mode": "needu"},
        "manifest": {
            "pipeline": {
                "response": config(response),
                "extraction": config(extraction),
            }
        },
    }


def test_model_metrics_attribute_each_role_and_keep_quantization_separate():
    runs = [run("r1"), run("r2")]
    runs[1]["manifest"]["pipeline"]["response"] = config("chat", "Q4")
    samples = [
        {
            "run_id": "r1",
            "stage": "turn",
            "input": "산책 좋아해요",
            "model_calls": [
                {"role": "response", "ok": True, "latency_ms": 100, "output": "어디로 가요?"},
                {
                    "role": "extraction",
                    "ok": False,
                    "latency_ms": 900,
                    "json_schema": {},
                    "error": "parse failed",
                },
                {"role": "persona", "ok": True, "latency_ms": 1, "output": "합성 사용자"},
                {"role": "response", "output": "in progress"},
            ],
        }
    ]
    models = model_evaluations(
        [{"id": "saved", **config("chat")}],
        runs,
        samples,
        [],
    )
    response = next(m for m in models if m["model"] == "chat" and m["precision"] == "BF16")
    extraction = next(m for m in models if m["model"] == "extract")
    quantized = next(m for m in models if m["precision"] == "Q4")
    assert response["attempted"] == 1
    assert response["roles"]["response"]["latency_p95_ms"] == 100
    assert response["response_style"]["question_rate"] == 1
    assert response["examples"]["response"]["input"] == "산책 좋아해요"
    assert extraction["success_rate"] == 0
    assert extraction["roles"]["extraction"]["structured_success_rate"] == 0
    assert extraction["roles"]["extraction"]["latency_p95_ms"] is None
    assert extraction["run_count"] == 2
    assert quantized["attempted"] == 0 and quantized["success_rate"] is None
    assert response["quality"]["exact_match_rate"] is None
    assert all("persona" not in m["roles"] for m in models)


def test_chat_results_and_quality_have_separate_denominators():
    quality_run = {
        "id": "q",
        "created_at": "2026-10-13",
        "status": "completed",
        "spec": {"mode": "quality"},
        "manifest": {},
        "connection": config("model"),
    }
    samples = [
        {
            "run_id": "q",
            "ok": True,
            "latency_ms": 30,
            "output": "{}",
            "evaluation": {"schema_valid": True, "exact_match": False},
        },
        {
            "run_id": "q",
            "ok": False,
            "error": "bad JSON",
            "evaluation": {"schema_valid": False, "exact_match": False},
        },
    ]
    chat = {
        "id": "c",
        "updated_at": "2026-10-14",
        "status": "closed",
        "config": {"response": config("model"), "extraction": config("model")},
        "traces": [
            {
                "calls": [
                    {"role": "response", "ok": True, "latency_ms": 10, "output": "안녕"},
                    {
                        "role": "response",
                        "ok": True,
                        "latency_ms": 50,
                        "output": "좋아요?",
                        "messages": [{"role": "user", "content": "산책"}],
                    },
                    {
                        "role": "response",
                        "ok": True,
                        "latency_ms": 70,
                        "output": "취향 요약",
                        "messages": [
                            {"role": "system", "content": FRIEND_SUMMARY_SYSTEM},
                            {"role": "user", "content": "{}"},
                        ],
                    },
                    {
                        "role": "extraction",
                        "ok": True,
                        "latency_ms": 20,
                        "output": {},
                        "json_schema": {},
                    },
                ]
            }
        ],
    }
    model = model_evaluations([], [quality_run], samples, [chat])[0]
    assert model["attempted"] == 6
    assert model["success_rate"] == pytest.approx(5 / 6)
    assert model["chat_count"] == 1
    assert model["quality"] == {"count": 2, "schema_rate": 0.5, "exact_match_rate": 0}
    assert model["roles"]["extraction"]["structured_success_rate"] == 1
    assert model["response_style"]["count"] == 1


def test_service_metrics_use_completed_reviews_and_weight_actual_conditions():
    runs = [run("a"), run("b"), run("pending", status="running")]
    for item, matched, total in zip(runs, [1, 9, 100], [2, 10, 100], strict=True):
        item["summary"] = {
            "query_review": {
                "score": matched / total,
                "matched_conditions": matched,
                "total_conditions": total,
                "exclusions": [{"actual": "안됨" if item["id"] == "a" else None}],
                "forbidden_queries": [{"query": "주의"}],
            }
        }
    samples = [
        {
            "run_id": "a",
            "stage": "turn",
            "ok": True,
            "latency_ms": 100,
            "pipeline": {"extraction_failed": True},
        },
        {"run_id": "b", "stage": "turn", "ok": True, "latency_ms": 200, "pipeline": {}},
        {"run_id": "a", "stage": "opening", "ok": True, "latency_ms": 999},
    ]
    group = service_evaluations(runs, samples)[0]
    assert group["reviewed_runs"] == 2
    assert group["condition_rate"] == pytest.approx(10 / 12)
    assert group["exclusion_rate"] == 0.5
    assert group["forbidden_count"] == 2
    assert group["extraction_failure_rate"] == 0.5
    assert group["turn_latency_p95_ms"] == 195


def test_overview_returns_saved_results_without_model_requests(tmp_path):
    def no_model_request(request):
        raise AssertionError("Overview must not invoke a model")

    app = create_app(tmp_path / "lab.sqlite3", transport=httpx.MockTransport(no_model_request))

    async def exercise():
        async with app.router.lifespan_context(app):
            store = app.state.store
            saved = store.create_run(
                {"mode": "load"},
                config("observed"),
                {},
            )
            store.sample(saved["id"], 0, {"ok": True, "output": "안녕", "latency_ms": 120})
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/overview")
                assert response.status_code == 200
                models = response.json()["model_evaluations"]
                observed = next(m for m in models if m["model"] == "observed")
                assert observed["attempted"] == 1
                assert observed["roles"]["response"]["latency_p95_ms"] == 120
                assert "service_evaluations" in response.json()

    asyncio.run(exercise())
