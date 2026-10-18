import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from admin_api.main import create_app
from admin_api.models import RunSpec
from admin_api.service_config import pipeline_configuration, service_configuration
from admin_api.store import Store
from app.application.conversation_prompts import SYSTEM_COMPANION
from app.core.config import Settings
from app.domain.recommendation import CatalogProduct, CatalogSearchResult, ProductKey
from benchmarks.needu import execute_needu


def settings(**overrides):
    values = dict(
        response_model_base_url="http://test/response/v1",
        response_model_name="response-real",
        extraction_model_base_url="http://test/extract/v1",
        extraction_model_name="extract-real",
        judgment_model_base_url=None,
        embedding_base_url="http://test/embedding/v1",
        embedding_dimensions=2,
        catalog_database_url=None,
        catalog_database_url_ro=None,
        model_api_key="synthetic-model-key",
        upstage_api_key="synthetic-embedding-key",
    )
    return Settings(_env_file=None, **(values | overrides))


def extraction(prompt):
    utterance = prompt.split("[이번 사용자 발화]\n")[-1]
    field = (
        "dislikes"
        if "강한 향" in utterance
        else "preferences"
        if "베이지색" in utterance
        else "hobbies"
    )
    value = "강한 향" if field == "dislikes" else "베이지색" if field == "preferences" else "캠핑"
    return {
        "items": [
            {
                "field": field,
                "value": value,
                "confidence": 0.95,
                "evidence": utterance,
                "evidenceType": "explicit",
                "intentType": None,
                "deferralReason": None,
                "aspect": "attribute" if field == "preferences" else None,
                "target": "캠핑" if field == "preferences" else None,
            }
        ],
        "axes": [],
        "drop": [],
    }


def handler(seen, broken_extraction=False):
    def handle(request):
        data = json.loads(request.content)
        seen.append((request.url.path, data))
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": i, "embedding": [0.0, 1.0] if "강한 향" in text else [1.0, 0.0]}
                        for i, text in enumerate(data["input"])
                    ]
                },
            )
        if "response_format" in data:
            if broken_extraction:
                return httpx.Response(503)
            output = json.dumps(extraction(data["messages"][-1]["content"]), ensure_ascii=False)
        else:
            output = "캠핑하며 쉬는 시간이 좋겠어요. 어떤 장비를 쓰세요?"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": output}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    return handle


def enqueue(store, configured, **overrides):
    spec = RunSpec(
        mode="needu",
        prompt="주말마다 캠핑을 가요.\n캠핑 장비는 베이지색이 좋아요.\n강한 향은 싫어요.",
        **overrides,
    )
    config = pipeline_configuration(configured, spec, store)
    store.create_run(spec.model_dump(), {"model": "service"}, {"pipeline": config})
    return store.claim()


def test_service_configuration_hides_secrets_and_selects_actual_prompt():
    configured = settings(catalog_database_url="postgresql://user:db-secret@host/catalog")
    result = service_configuration(configured)
    serialized = json.dumps(result, ensure_ascii=False)
    assert "synthetic-model-key" not in serialized
    assert "synthetic-embedding-key" not in serialized
    assert "db-secret" not in serialized
    assert "postgresql://" not in serialized
    assert "conversation_style" not in result
    assert result["opening_messages"][0]["content"] == SYSTEM_COMPANION
    assert "items" in result["extraction_prompts"]["direct"]["schema"]["properties"]


def test_full_conversation_uses_service_prompts_schema_and_query_builder(tmp_path):
    store = Store(tmp_path / "lab.db")
    configured = settings()
    run = enqueue(store, configured)
    seen = []

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler(seen))) as client:
            await execute_needu(store, run, settings=configured, client=client)

    asyncio.run(exercise())
    result = store.run(run["id"])
    assert result["status"] == "completed"
    assert result["summary"]["executed_turns"] == 3
    response_requests = [data for path, data in seen if "/response/" in path]
    assert response_requests[0]["messages"][0]["content"] == SYSTEM_COMPANION
    assert all(data["model"] == "response-real" for data in response_requests)
    assert all(
        data["max_tokens"] == configured.response_model_max_tokens for data in response_requests
    )
    assert "주말마다 캠핑을 가요." in json.dumps(response_requests[2], ensure_ascii=False)
    extracted = [data for path, data in seen if "/extract/" in path]
    assert "items" in extracted[0]["response_format"]["json_schema"]["schema"]["properties"]
    assert all(data["model"] == "extract-real" for data in extracted)
    final = result["samples"][-1]["pipeline"]
    assert "캠핑 활동에서 사용하는 제품" in [q["text"] for q in final["queries"]]
    assert final["exclusions"] == ["강한 향"]
    assert any(i["kind"] == "exclusion" for i in final["embedding_inputs"])
    assert final["embedding_status"] == "not_requested"
    assert not any(path.endswith("/embeddings") for path, _ in seen)
    assert all("synthetic-model-key" not in json.dumps(sample) for sample in result["samples"])


def test_role_overrides_route_same_model_name_to_separate_endpoints(tmp_path):
    store = Store(tmp_path / "lab.db")
    configured = settings()
    for role, path in [("response", "response"), ("extract", "extract")]:
        store.save_connection(
            role,
            {
                "name": role,
                "provider": "lmstudio",
                "environment": "local",
                "base_url": f"http://test/{path}/v1",
                "model": "same-local-model",
                "api_key_env": "",
                "revision": "",
                "precision": "Q4",
                "hardware": "local",
            },
        )
    run = enqueue(
        store,
        configured,
        response_connection_id="response",
        extraction_connection_id="extract",
    )
    seen = []

    def check(request):
        assert "Authorization" not in request.headers
        return handler(seen)(request)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(check)) as client:
            await execute_needu(store, run, settings=configured, client=client)

    asyncio.run(exercise())
    assert store.run(run["id"])["status"] == "completed"
    assert all(data["model"] == "same-local-model" for _, data in seen)
    assert seen[0][1]["messages"][0]["content"] == SYSTEM_COMPANION
    assert any("/extract/" in path for path, _ in seen)


class Catalog:
    def __init__(self):
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["vector"] == [0.0, 1.0]:
            return []
        return [
            CatalogSearchResult(
                CatalogProduct(ProductKey("test", "123"), "베이지 캠핑 머그", Decimal("20000")), 0.9
            )
        ]


@pytest.mark.parametrize("stage", ["embedding", "recommendation"])
def test_embedding_and_real_ranking_use_exact_service_inputs(tmp_path, stage):
    store = Store(tmp_path / "lab.db")
    configured = settings(catalog_database_url_ro="postgresql://reader:secret@host/catalog")
    run = enqueue(store, configured, recommendation_stage=stage)
    seen = []
    catalog = Catalog()

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler(seen))) as client:
            await execute_needu(store, run, settings=configured, client=client, catalog=catalog)

    asyncio.run(exercise())
    saved = store.run(run["id"])
    assert saved["status"] == "completed"
    final = saved["samples"][-1]["pipeline"]
    embedded = next(data["input"] for path, data in seen if path.endswith("/embeddings"))
    assert embedded == [item["text"] for item in final["embedding_inputs"]]
    assert final["embedding_status"] == "completed"
    assert all(vector["dimensions"] == 2 for vector in final["embedding_batches"][0]["vectors"])
    if stage == "recommendation":
        assert catalog.calls
        assert final["recommendations"]["self"]["items"][0]["externalId"] == "123"
        assert all(
            call["embedding_space_id"] == configured.embedding_space_id for call in catalog.calls
        )
    else:
        assert not catalog.calls
        assert "recommendations" not in final


def test_failed_extraction_keeps_service_fallback_visible(tmp_path):
    store = Store(tmp_path / "lab.db")
    configured = settings()
    run = enqueue(store, configured)
    seen = []

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler(seen, True))) as client:
            await execute_needu(store, run, settings=configured, client=client)

    asyncio.run(exercise())
    saved = store.run(run["id"])
    assert saved["status"] == "failed"
    assert saved["summary"]["extraction_failed_turns"] == 3
    assert saved["samples"][1]["output"]
    assert not saved["samples"][1]["ok"]
    assert saved["samples"][1]["pipeline"]["extraction_failed"]
    assert saved["samples"][1]["pipeline"]["queries"] == []
    assert (
        next(c for c in saved["samples"][1]["model_calls"] if c["role"] == "extraction")[
            "http_status"
        ]
        == 503
    )


def test_changed_service_environment_stops_before_model_call(tmp_path):
    store = Store(tmp_path / "lab.db")
    run = enqueue(store, settings())
    changed = settings(response_model_name="response-changed")
    asyncio.run(execute_needu(store, run, settings=changed))
    assert store.run(run["id"])["status"] == "failed"
    assert store.run(run["id"])["samples"] == []


def test_environment_judgment_path_uses_candidate_schema_and_systemone(tmp_path):
    store = Store(tmp_path / "lab.db")
    configured = settings(
        judgment_model_base_url="http://test/judge/v1",
        openrouter_api_key="synthetic-openrouter-key",
    )
    spec = RunSpec(mode="needu", prompt="주말마다 캠핑을 가요.")
    config = pipeline_configuration(configured, spec, store)
    store.create_run(spec.model_dump(), {"model": "service"}, {"pipeline": config})
    run = store.claim()
    seen = []

    def handle(request):
        data = json.loads(request.content)
        seen.append((request.url.path, data))
        expected_key = (
            "synthetic-openrouter-key"
            if request.url.path.endswith("/systemone")
            else "synthetic-model-key"
        )
        assert request.headers["Authorization"] == f"Bearer {expected_key}"
        if request.url.path.endswith("/systemone"):
            assert "stance_0" in data["questions"]
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "stance_0": {"type": "choice", "choice": "like", "confidence": 1.0},
                        "explicit_0": {"type": "noul", "noul": 0.95},
                        "habitual_0": {"type": "noul", "noul": 0.95},
                        "evidence_0": {"type": "choice", "choice": "s0", "confidence": 1.0},
                    }
                },
            )
        if "response_format" in data:
            assert "candidates" in data["response_format"]["json_schema"]["schema"]["properties"]
            output = json.dumps({"candidates": [{"subject": "캠핑", "kind": "activity"}]})
        else:
            output = "캠핑에서 어떤 시간을 좋아하세요?"
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await execute_needu(store, run, settings=configured, client=client)

    asyncio.run(exercise())
    result = store.run(run["id"])
    assert result["status"] == "completed"
    assert any(path.endswith("/systemone") for path, _ in seen)
    assert result["samples"][1]["pipeline"]["profile"]["hobbies"][0]["value"] == "캠핑"
    assert any(call["role"] == "judgment" for call in result["samples"][1]["model_calls"])


def test_cancellation_closes_parallel_service_model_calls(tmp_path):
    store = Store(tmp_path / "lab.db")
    configured = settings()
    run = enqueue(store, configured)
    started = asyncio.Event()
    pending = 0
    aborted = []
    seen = []

    async def handle(request):
        nonlocal pending
        if not seen:
            return handler(seen)(request)  # Allow the greeting to finish first.
        pending += 1
        if pending == 2:
            started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            aborted.append(request.url.path)
            raise
        return handler(seen)(request)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            task = asyncio.create_task(
                execute_needu(store, run, settings=configured, client=client)
            )
            await asyncio.wait_for(started.wait(), 2)
            store.cancel(run["id"])
            await asyncio.wait_for(task, 2)

    asyncio.run(exercise())
    assert store.run(run["id"])["status"] == "cancelled"
    assert len(aborted) == 2
    assert len(store.run(run["id"])["samples"]) == 2


def test_api_service_run_snapshots_environment_and_overrides(tmp_path):
    app = create_app(tmp_path / "lab.db", service_settings=settings())

    async def exercise():
        async with app.router.lifespan_context(app):
            app.state.store.beat()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/runs",
                    json={
                        "mode": "needu",
                        "prompt": "캠핑을 좋아해요.\n커피도 좋아요.",
                        "judgment_mode": "disabled",
                    },
                )
                assert response.status_code == 201
                body = response.json()
                assert body["spec"]["requests"] == 4
                assert "conversation_style" not in body["manifest"]["pipeline"]
                assert not body["manifest"]["pipeline"]["judgment_enabled"]
                assert "synthetic-model-key" not in response.text
                profile = await client.get("/api/service-config")
                assert profile.status_code == 200
                assert "synthetic-embedding-key" not in profile.text
                too_long = await client.post(
                    "/api/runs", json={"mode": "needu", "prompt": "턴\n" * 21}
                )
                assert too_long.status_code == 422
                missing_db = await client.post(
                    "/api/runs",
                    json={
                        "mode": "needu",
                        "prompt": "캠핑 좋아요.",
                        "recommendation_stage": "recommendation",
                    },
                )
                assert missing_db.status_code == 422

    asyncio.run(exercise())
