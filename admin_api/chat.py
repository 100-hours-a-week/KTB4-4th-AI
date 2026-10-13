"""Interactive Need U sessions persisted only in the local lab database."""

import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from admin_api.models import RunSpec
from admin_api.service_config import environment_hash, pipeline_configuration
from admin_api.store import now
from app.application.chat_use_cases import ChatUseCaseError, ProfileTooSparseError
from app.domain.conversation.models import ConversationState
from benchmarks.needu import build_chat, profile_snapshot
from benchmarks.personas import persona_snapshot


class LocalChatStates:
    def __init__(self, store, session_id):
        self.store, self.session_id = store, session_id

    async def load(self, session_id):
        session = self.store.chat(self.session_id)
        return session.get("state") if session else None

    async def save(self, session_id, state, *, ttl_seconds):
        session = self.store.chat(self.session_id)
        session["state"] = dict(state)
        self.store.save_chat(session)

    async def delete(self, session_id):
        session = self.store.chat(self.session_id)
        existed = bool(session.get("state"))
        session["state"] = None
        self.store.save_chat(session)
        return existed


def public_chat(session):
    return {key: value for key, value in session.items() if key != "state"}


class LabChats:
    def __init__(self, store, client, settings, manifest):
        self.store, self.client = store, client
        self.settings, self.manifest = settings, manifest
        self.locks = {}
        self.slots = asyncio.Semaphore(2)

    def get(self, session_id):
        session = self.store.chat(session_id)
        if not session:
            raise HTTPException(404, "대화 세션을 찾을 수 없습니다.")
        return session

    @asynccontextmanager
    async def operation(self, session_id):
        lock = self.locks.setdefault(session_id, asyncio.Lock())
        if lock.locked():
            raise HTTPException(409, "이 대화의 요청이 진행 중입니다.")
        if self.slots.locked():
            raise HTTPException(429, "동시 대화 요청은 최대 2개입니다. 잠시 후 다시 시도하세요.")
        async with lock, self.slots:
            session = self.get(session_id)
            settings = self.settings()
            config = session["config"]
            if session["manifest"]["source_hash"] != self.manifest["source_hash"] or config[
                "environment_hash"
            ] != environment_hash(settings):
                raise HTTPException(
                    409, "코드 또는 환경 설정이 변경됐습니다. 새 대화를 시작하세요."
                )
            calls = []
            started = time.perf_counter()
            try:
                chat, states = build_chat(
                    config, settings, self.client, calls, LocalChatStates(self.store, session_id)
                )
                async with asyncio.timeout(300):
                    yield session, chat, states, calls
            except ProfileTooSparseError:
                self.record(
                    session_id,
                    calls,
                    started,
                    error="분석할 취향 근거가 부족합니다.",
                    status="expired",
                )
                raise HTTPException(
                    422, "취향 근거가 부족해 분석할 수 없습니다. 새 대화를 시작하세요."
                ) from None
            except ChatUseCaseError as error:
                self.record(session_id, calls, started, error=type(error).__name__)
                raise HTTPException(
                    409, "대화를 진행할 수 없습니다. 세션 상태와 모델 연결을 확인하세요."
                ) from None
            except HTTPException:
                raise
            except Exception:
                self.record(
                    session_id, calls, started, error="모델 호출 실패. 연결과 설정을 확인하세요."
                )
                raise HTTPException(
                    502, "모델 호출 실패. 연결·타임아웃·원본 호출 기록을 확인하세요."
                ) from None
            else:
                self.record(session_id, calls, started)

    def record(self, session_id, calls, started, error=None, status=None):
        session = self.get(session_id)
        session["updated_at"] = now()
        session["error"] = error
        session["traces"].append(
            jsonable_encoder(
                {
                    "calls": calls,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error": error,
                }
            )
        )
        if session.get("state"):
            state = ConversationState.from_dict(session["state"])
            session.update(status=state.status.value, turn_count=state.turn_count)
            session["messages"] = [turn.to_dict() for turn in state.history]
            session["pipeline"] = jsonable_encoder(profile_snapshot(state))
        if status:
            session["status"] = status
        self.store.save_chat(session)

    async def create(self, spec):
        persona = persona_snapshot(spec.persona_id) if spec.persona_id else None
        if spec.persona_id and persona is None:
            raise HTTPException(422, "페르소나를 찾을 수 없습니다.")
        config = pipeline_configuration(
            self.settings(),
            RunSpec(prompt="대화 세션", **spec.model_dump(exclude={"title", "persona_id"})),
            self.store,
        )
        session = {
            "id": uuid4().hex,
            "title": spec.title,
            "config": config,
            "persona": persona,
            "manifest": self.manifest,
            "created_at": now(),
            "updated_at": now(),
            "status": "starting",
            "turn_count": 0,
            "messages": [],
            "traces": [],
            "pipeline": {},
            "error": None,
        }
        self.store.save_chat(session)
        try:
            async with self.operation(session["id"]) as (_, chat, _, _):
                await chat.create_session(user_id=1, conversation_room_id=1)
        except Exception:
            failed = self.get(session["id"])
            failed["status"] = "failed"
            self.store.save_chat(failed)
            raise
        return public_chat(self.get(session["id"]))

    async def send(self, session_id, message):
        async with self.operation(session_id) as (_, chat, states, _):
            if await states.load(1) is None:
                session = self.get(session_id)
                session["status"] = "expired"
                self.store.save_chat(session)
                raise HTTPException(409, "대화 세션이 만료됐습니다. 새 대화를 시작하세요.")
            result = await chat.send_message(1, message, user_id=1)
            if result.extraction_failed:
                session = self.get(session_id)
                session["last_extraction_failed"] = True
                self.store.save_chat(session)
            else:
                session = self.get(session_id)
                session["last_extraction_failed"] = False
                self.store.save_chat(session)
        return public_chat(self.get(session_id))

    async def analyze(self, session_id):
        async with self.operation(session_id) as (_, chat, _, _):
            analysis = await chat.analyze_session(1)
            await chat.close_session(1, user_id=1)
            session = self.get(session_id)
            session["analysis"] = {
                "summary": analysis.summary,
                "pipeline": jsonable_encoder(profile_snapshot(analysis.state, analysis)),
            }
            self.store.save_chat(session)
        return public_chat(self.get(session_id))

    def delete(self, session_id):
        self.get(session_id)
        if self.locks.get(session_id) and self.locks[session_id].locked():
            raise HTTPException(409, "요청이 끝난 뒤 대화를 삭제하세요.")
        self.store.delete_chat(session_id)
        self.locks.pop(session_id, None)
