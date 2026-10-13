import asyncio
import json

import httpx

from admin_api.main import create_app
from app.core.config import Settings
from benchmarks.personas import PERSONAS, persona_snapshot


def configured():
    return Settings(
        _env_file=None,
        response_model_base_url="http://test/response/v1",
        response_model_name="response-model",
        extraction_model_base_url="http://test/extraction/v1",
        extraction_model_name="extraction-model",
        judgment_model_base_url=None,
        conversation_style="companion",
        model_api_key="test-secret",
    )


def model_handler(seen, broken=False):
    def handle(request):
        data = json.loads(request.content)
        seen.append((request, data))
        if broken:
            return httpx.Response(503, text="secret-provider-error")
        if "response_format" in data:
            utterance = data["messages"][-1]["content"].split("[이번 사용자 발화]\n")[-1]
            output = json.dumps(
                {
                    "items": [
                        {
                            "field": "hobbies",
                            "value": "캠핑",
                            "confidence": 0.95,
                            "evidence": utterance,
                            "evidenceType": "explicit",
                            "intentType": None,
                            "deferralReason": None,
                            "aspect": None,
                            "target": None,
                        }
                    ],
                    "axes": [],
                    "drop": [],
                    "noneAnswer": False,
                },
                ensure_ascii=False,
            )
        else:
            output = "어떤 이야기를 나누고 싶어요?"
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    return handle


def test_interactive_chat_uses_real_service_and_persists_history_after_restart(tmp_path):
    seen = []
    path = tmp_path / "lab.db"
    app = create_app(
        path, transport=httpx.MockTransport(model_handler(seen)), service_settings=configured()
    )

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                await client.put(
                    "/api/connections/ollama",
                    json={
                        "name": "Local Solar",
                        "provider": "ollama",
                        "base_url": "http://local/v1",
                        "model": "solar:latest",
                    },
                )
                created = await client.post(
                    "/api/chats",
                    json={
                        "title": "페르소나 검증",
                        "response_connection_id": "ollama",
                        "judgment_mode": "disabled",
                        "persona_id": "p01",
                    },
                )
                assert created.status_code == 201, created.text
                session = created.json()
                sid = session["id"]
                assert session["persona"]["name"] == "김소연"
                assert session["status"] == "active"
                assert "state" not in session
                for text in ["주말마다 캠핑을 가요.", "캠핑 가서 커피 내리는 시간이 좋아요."]:
                    response = await client.post(
                        f"/api/chats/{sid}/messages", json={"userId": 1, "message": text}
                    )
                    assert response.status_code == 200, response.text
                latest = response.json()
                assert latest["turn_count"] == 2
                assert len(latest["messages"]) == 5
                assert any("캠핑" in q["text"] for q in latest["pipeline"]["queries"])
                response_calls = [data for _, data in seen if "response_format" not in data]
                assert "주말마다 캠핑을 가요." in json.dumps(response_calls[-1], ensure_ascii=False)
                assert response_calls[-1]["model"] == "solar:latest"
                assert all(
                    "Authorization" not in req.headers
                    for req, data in seen
                    if data["model"] == "solar:latest"
                )
                assert "test-secret" not in response.text
                assert (
                    await client.post(
                        f"/api/chats/{sid}/messages", json={"userId": 1, "message": "a" * 501}
                    )
                ).status_code == 422
                assert (await client.post(f"/api/chats/{sid}/analysis")).status_code == 200
                assert (
                    await client.post(
                        f"/api/chats/{sid}/messages", json={"userId": 1, "message": "다음 말"}
                    )
                ).status_code == 409
                return sid

    sid = asyncio.run(exercise())
    restored_app = create_app(path, service_settings=configured())

    async def restored():
        async with restored_app.router.lifespan_context(restored_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=restored_app), base_url="http://testserver"
            ) as client:
                session = (await client.get(f"/api/chats/{sid}")).json()
                assert session["status"] == "closed"
                assert len(session["messages"]) == 5
                assert session["analysis"]["pipeline"]["queries"]
                assert (await client.delete(f"/api/chats/{sid}")).status_code == 200
                assert (await client.get(f"/api/chats/{sid}")).status_code == 404

    asyncio.run(restored())


def test_model_failure_is_recorded_without_provider_body_or_credentials(tmp_path):
    app = create_app(
        tmp_path / "lab.db",
        transport=httpx.MockTransport(model_handler([], broken=True)),
        service_settings=configured(),
    )

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                assert (await client.post("/api/chats", json={})).status_code == 409
                rows = (await client.get("/api/chats")).json()
                assert rows[0]["status"] == "failed"
                detail = (await client.get(f"/api/chats/{rows[0]['id']}")).json()
                assert detail["traces"][0]["calls"][0]["http_status"] == 503
                assert "secret-provider-error" not in json.dumps(detail)
                assert "test-secret" not in json.dumps(detail)

    asyncio.run(exercise())


def test_personas_are_valid_service_inputs_and_recorded_in_run_manifest(tmp_path):
    app = create_app(tmp_path / "lab.db", service_settings=configured())

    async def exercise():
        async with app.router.lifespan_context(app):
            app.state.store.beat()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                personas = (await client.get("/api/personas")).json()["personas"]
                assert len(personas) == 10
                for persona in personas:
                    response = await client.post(
                        "/api/runs",
                        json={
                            "mode": "needu",
                            "persona_id": persona["id"],
                            "prompt": "\n".join(persona["turns"]),
                        },
                    )
                    assert response.status_code == 201, response.text
                    assert response.json()["manifest"]["persona"]["expected"] == persona["expected"]
                snapshot = persona_snapshot("p01")
                snapshot["turns"][0] = "changed"
                assert PERSONAS[0]["turns"][0] != "changed"

    asyncio.run(exercise())


def test_chat_blocks_changed_configuration_without_issuing_model_calls(tmp_path):
    seen = []
    settings = configured()
    app = create_app(
        tmp_path / "lab.db",
        transport=httpx.MockTransport(model_handler(seen)),
        service_settings=settings,
    )

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                session = (await client.post("/api/chats", json={})).json()
                settings.response_model_name = "changed"
                assert (
                    await client.post(
                        f"/api/chats/{session['id']}/messages",
                        json={"userId": 1, "message": "캠핑 좋아요"},
                    )
                ).status_code == 409
                assert len(seen) == 1

    asyncio.run(exercise())


def test_overlapping_messages_are_rejected_without_duplicate_turns(tmp_path):
    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        blocked = False
        handle = model_handler([])

        async def slow_model(request):
            if blocked:
                started.set()
                await release.wait()
            return handle(request)

        app = create_app(
            tmp_path / "lab.db",
            transport=httpx.MockTransport(slow_model),
            service_settings=configured(),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                sid = (await client.post("/api/chats", json={})).json()["id"]
                blocked = True
                body = {"userId": 1, "message": "주말마다 캠핑을 가요."}
                pending = asyncio.create_task(client.post(f"/api/chats/{sid}/messages", json=body))
                await asyncio.wait_for(started.wait(), 2)
                assert (
                    await client.post(f"/api/chats/{sid}/messages", json=body)
                ).status_code == 409
                release.set()
                result = await asyncio.wait_for(pending, 2)
                assert result.status_code == 200
                assert result.json()["turn_count"] == 1
                assert len(result.json()["messages"]) == 3

    asyncio.run(exercise())


def test_expired_chat_keeps_transcript_but_does_not_call_model(tmp_path):
    seen = []
    app = create_app(
        tmp_path / "lab.db",
        transport=httpx.MockTransport(model_handler(seen)),
        service_settings=configured(),
    )

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                sid = (await client.post("/api/chats", json={})).json()["id"]
                saved = app.state.store.chat(sid)
                saved["state"]["created_at"] = "2000-01-01T00:00:00+00:00"
                saved["state"]["last_active_at"] = "2000-01-01T00:00:00+00:00"
                app.state.store.save_chat(saved)
                response = await client.post(
                    f"/api/chats/{sid}/messages", json={"userId": 1, "message": "캠핑 좋아요"}
                )
                assert response.status_code == 409
                detail = (await client.get(f"/api/chats/{sid}")).json()
                assert detail["status"] == "expired"
                assert len(detail["messages"]) == 1
                assert len(seen) == 1

    asyncio.run(exercise())
