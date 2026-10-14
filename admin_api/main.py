import csv
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from admin_api.chat import LabChats, public_chat
from admin_api.client import probe, resolve_huggingface_model
from admin_api.model_evaluations import model_evaluations, service_evaluations
from admin_api.models import ChatSpec, Connection, RunSpec
from admin_api.provenance import code_revision, source_hash
from admin_api.service_config import (
    load_service_settings,
    pipeline_configuration,
    service_configuration,
)
from admin_api.store import ROOT, Store
from app.api.schemas.chat import ChatMessageRequest
from benchmarks.fixtures import CASES, SCHEMA, SYSTEM, VERSION, fixture_hash
from benchmarks.persona_dialogue import PROMPT_VERSION, persona_system
from benchmarks.personas import PERSONAS, persona_snapshot
from benchmarks.personas import VERSION as PERSONA_VERSION


def create_app(db_path: Path | None = None, transport=None, service_settings=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(db_path)
        app.state.manifest = {"code_revision": code_revision(), "source_hash": source_hash()}
        async with httpx.AsyncClient(
            transport=transport, timeout=None, follow_redirects=False
        ) as client:
            app.state.client = client
            app.state.chats = LabChats(
                app.state.store, client, settings_for_run, app.state.manifest
            )
            yield

    app = FastAPI(title="Need U AI Lab", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        origin = request.headers.get("origin")
        expected = str(request.base_url).rstrip("/")
        if origin and origin != expected:
            return Response("Cross-origin access denied", status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/overview")
    async def overview(request: Request):
        store = request.app.state.store
        connections, runs = store.connections(), store.runs()
        samples = store.overview_samples()
        return {
            "worker": store.worker_status(),
            "connections": connections,
            "runs": runs,
            "model_evaluations": model_evaluations(connections, runs, samples, store.chats()),
            "service_evaluations": service_evaluations(runs, samples),
        }

    @app.get("/api/connections")
    async def connections(request: Request):
        return request.app.state.store.connections()

    @app.post("/api/connections", status_code=201)
    async def create_connection(body: Connection, request: Request):
        return await save_connection(uuid4().hex, body, request)

    async def save_connection(connection_id, body, request):
        data = body.model_dump()
        result = None
        if body.provider == "huggingface":
            result = await probe(request.app.state.client, data)
            metadata = result.get("model_metadata")
            if metadata is None:
                raise HTTPException(502, result.get("error", "Hugging Face 목록 조회 실패"))
            if body.model.strip():
                try:
                    route = resolve_huggingface_model(body.model, metadata)
                except ValueError as error:
                    raise HTTPException(422, str(error)) from None
                data["model"] = route["id"]
                result["selected_route"] = route
        store = request.app.state.store
        store.save_connection(connection_id, data)
        if result:
            store.save_probe(connection_id, result)
        return store.connection(connection_id)

    @app.put("/api/connections/{connection_id}")
    async def update_connection(connection_id: str, body: Connection, request: Request):
        store = request.app.state.store
        if not store.connection(connection_id):
            raise HTTPException(404, "연결을 찾을 수 없습니다.")
        return await save_connection(connection_id, body, request)

    @app.post("/api/connections/{connection_id}/probe")
    async def probe_connection(connection_id: str, request: Request):
        store = request.app.state.store
        connection = store.connection(connection_id)
        if not connection:
            raise HTTPException(404, "연결을 찾을 수 없습니다.")
        result = await probe(request.app.state.client, connection)
        store.save_probe(connection_id, result)
        return result

    @app.get("/api/fixtures")
    async def fixtures():
        return {"version": VERSION, "hash": fixture_hash(), "cases": CASES, "system": SYSTEM}

    def settings_for_run():
        try:
            return service_settings or load_service_settings()
        except Exception:
            raise HTTPException(
                422, "니쥬 환경변수 설정을 읽을 수 없습니다. 설정 형식을 확인하세요."
            ) from None

    @app.get("/api/service-config")
    async def service_config():
        return service_configuration(settings_for_run())

    @app.get("/api/personas")
    async def personas():
        return {"version": PERSONA_VERSION, "personas": PERSONAS}

    @app.get("/api/chats")
    async def chats(request: Request):
        return [
            {
                key: value
                for key, value in public_chat(session).items()
                if key not in {"messages", "traces", "pipeline", "analysis"}
            }
            for session in request.app.state.store.chats()
        ]

    @app.post("/api/chats", status_code=201)
    async def create_chat(body: ChatSpec, request: Request):
        try:
            return await request.app.state.chats.create(body)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/chats/{session_id}")
    async def get_chat(session_id: str, request: Request):
        return public_chat(request.app.state.chats.get(session_id))

    @app.post("/api/chats/{session_id}/messages")
    async def send_chat_message(session_id: str, body: ChatMessageRequest, request: Request):
        return await request.app.state.chats.send(session_id, body.message)

    @app.post("/api/chats/{session_id}/analysis")
    async def analyze_chat(session_id: str, request: Request):
        return await request.app.state.chats.analyze(session_id)

    @app.delete("/api/chats/{session_id}")
    async def delete_chat(session_id: str, request: Request):
        request.app.state.chats.delete(session_id)
        return {"ok": True}

    @app.post("/api/runs", status_code=201)
    async def create_run(body: RunSpec, request: Request):
        store = request.app.state.store
        if not store.worker_status()["online"]:
            raise HTTPException(
                409, "테스트 워커가 꺼져 있습니다. python -m benchmarks.worker를 실행하세요."
            )
        if sum(r["status"] in {"queued", "running"} for r in store.runs()) >= 10:
            raise HTTPException(409, "대기 작업 상한(10개)에 도달했습니다.")
        spec = body.model_dump()
        manifest = {**request.app.state.manifest}
        if body.mode == "needu":
            try:
                config = pipeline_configuration(settings_for_run(), body, store)
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            manifest["pipeline"] = config
            if body.persona_id:
                persona = persona_snapshot(body.persona_id)
                if persona is None:
                    raise HTTPException(422, "페르소나를 찾을 수 없습니다.")
                manifest["persona"] = persona
                if body.dialogue_mode == "simulated":
                    manifest["persona_simulator"] = {
                        "prompt_version": PROMPT_VERSION,
                        "system": persona_system(persona),
                    }
            snapshot = {
                "id": "needu",
                "name": "니쥬 서비스",
                "provider": "service",
                "environment": "service",
                "model": f"{config['response']['model']} / {config['extraction']['model']}",
            }
            return store.create_run(spec, snapshot, manifest)
        connection = store.connection(body.connection_id)
        if not connection:
            raise HTTPException(404, "연결을 찾을 수 없습니다.")
        if not connection["model"].strip():
            raise HTTPException(422, "연결 관리에서 실제 모델 ID를 입력하고 저장하세요.")
        if body.mode == "quality":
            spec["requests"] = len(CASES)
            spec["system"] = SYSTEM
            spec["prompt"] = ""
        snapshot = {k: v for k, v in connection.items() if k not in {"probe", "checked_at"}}
        manifest = {
            **manifest,
            "fixture_version": VERSION,
            "fixture_hash": fixture_hash(),
        }
        if body.mode == "quality":
            manifest["fixture"] = {"cases": CASES, "schema": SCHEMA, "system": SYSTEM}
        return store.create_run(spec, snapshot, manifest)

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str, request: Request):
        run = request.app.state.store.run(run_id)
        if not run:
            raise HTTPException(404, "실험을 찾을 수 없습니다.")
        return run

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, request: Request):
        store = request.app.state.store
        if not store.run(run_id):
            raise HTTPException(404, "실험을 찾을 수 없습니다.")
        store.cancel(run_id)
        return {"ok": True}

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str, request: Request):
        store = request.app.state.store
        if not store.run(run_id):
            raise HTTPException(404, "실험을 찾을 수 없습니다.")
        if not store.delete(run_id):
            raise HTTPException(409, "실행 중인 작업은 취소 완료 후 삭제하세요.")
        return {"ok": True}

    @app.get("/api/runs/{run_id}/export")
    async def export_run(run_id: str, request: Request, format: str = "json"):
        run = request.app.state.store.run(run_id)
        if not run:
            raise HTTPException(404, "실험을 찾을 수 없습니다.")
        if format not in {"json", "csv"}:
            raise HTTPException(422, "json 또는 csv 형식만 지원합니다.")
        if format == "json":
            data = json.dumps(run, ensure_ascii=False, indent=2)
            media = "application/json"
        else:
            buffer = io.StringIO()
            fields = [
                "run_id",
                "ordinal",
                "label",
                "ok",
                "latency_ms",
                "first_chunk_ms",
                "completion_tokens",
                "schema_valid",
                "exact_match",
                "error",
            ]
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            for sample in run["samples"]:
                row = {key: sample.get(key, "") for key in fields}
                row.update(
                    run_id=run_id,
                    completion_tokens=(sample.get("usage") or {}).get("completion_tokens", ""),
                )
                row.update(
                    {
                        key: sample.get("evaluation", {}).get(key, "")
                        for key in ["schema_valid", "exact_match"]
                    }
                )
                # Spreadsheet programs must not execute provider-derived strings as formulas.
                for key, value in row.items():
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                        row[key] = "'" + value
                writer.writerow(row)
            data = "\ufeff" + buffer.getvalue()
            media = "text/csv"
        return Response(
            data,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="ai-lab-{run_id}.{format}"'},
        )

    @app.get("/api/colab/notebook", include_in_schema=False)
    async def colab_notebook():
        return FileResponse(
            ROOT / "notebooks/ai_lab_colab.ipynb",
            media_type="application/x-ipynb+json",
            filename="ai_lab_colab.ipynb",
        )

    app.mount("/assets", StaticFiles(directory=ROOT / "dashboard"), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "dashboard/index.html")

    return app


app = create_app()
