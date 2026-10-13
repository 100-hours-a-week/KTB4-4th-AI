"""Run the production Need U use cases with isolated experiment sessions."""

import asyncio
import time
from math import sqrt

import httpx
from fastapi.encoders import jsonable_encoder

from admin_api.service_config import environment_hash, load_service_settings, model_key
from app.api.routes.v1_chat import recommendation_profile
from app.application.chat_use_cases import ChatUseCases, ProfileAnalysis
from app.application.conversation_service import ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.recommendation_engine import RecommendationEngine
from app.application.recommendation_service import V1RecommendationService, recommendation_inputs
from app.domain.conversation.models import SessionStatus
from app.domain.conversation.policy import recommendation_readiness
from app.domain.recommendation import build_search_queries
from app.infrastructure.embedding import OpenAICompatibleEmbedder
from app.infrastructure.model_gateway import OpenAICompatibleModelGateway, SystemOneJudgmentGateway
from app.infrastructure.persistence import InMemorySessionStore
from benchmarks.persona_dialogue import PersonaReply, persona_messages
from benchmarks.query_review import review_queries
from benchmarks.runner import percentile, summarize


class RecordingGateway(OpenAICompatibleModelGateway):
    def __init__(self, client, config, settings, calls, role):
        self.config = config
        self.calls = calls
        self.role = role
        self.metadata = None
        self.raw_output = None
        super().__init__(
            client=client,
            model_base_urls={config["model"]: config["base_url"]},
            model_max_tokens={config["model"]: config["max_tokens"]},
            model_temperatures={config["model"]: config["temperature"]}
            if "temperature" in config
            else None,
            api_key=model_key(config, settings),
        )

    async def _post(self, model, payload):
        body = await super()._post(model, payload)
        self.metadata = body.get("usage")
        try:
            self.raw_output = body["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError):
            self.raw_output = None
        return body

    async def _record(self, messages, schema=None):
        started = time.perf_counter()
        call = {
            "role": self.role,
            "model": self.config["model"],
            "messages": list(messages),
            "json_schema": schema,
            "max_tokens": self.config["max_tokens"],
        }
        self.calls.append(call)
        self.metadata = None
        self.raw_output = None
        try:
            if schema is None:
                result = await super().complete(messages, model=self.config["model"])
            else:
                result = await super().structured(
                    messages, model=self.config["model"], json_schema=schema
                )
            call.update(ok=True, output=result)
            return result
        except BaseException as error:
            call.update(ok=False, error=f"{type(error).__name__}: 모델 호출 또는 파싱 실패")
            cause = error.__cause__
            if isinstance(cause, httpx.HTTPStatusError):
                call["http_status"] = cause.response.status_code
            raise
        finally:
            call.update(
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=self.metadata,
                raw_output=self.raw_output,
            )

    async def complete(self, messages, *, model):
        return await self._record(messages)

    async def structured(self, messages, *, model, json_schema):
        return await self._record(messages, json_schema)


class RoleGateway:
    def __init__(self, gateways):
        self.gateways = gateways

    async def complete(self, messages, *, model):
        return await self.gateways[model].complete(messages, model=model)

    async def structured(self, messages, *, model, json_schema):
        return await self.gateways[model].structured(messages, model=model, json_schema=json_schema)

    async def stream(self, messages, *, model):
        gateway = self.gateways[model]
        async for text in gateway.stream(messages, model=gateway.config["model"]):
            yield text


class RecordingJudgment:
    def __init__(self, client, config, settings, calls):
        self.calls = calls
        self.gateway = SystemOneJudgmentGateway(
            client=client, base_url=config["base_url"], api_key=model_key({}, settings)
        )

    async def decide(self, state, questions, *, model):
        started = time.perf_counter()
        call = {"role": "judgment", "model": model, "state": state, "questions": questions}
        self.calls.append(call)
        try:
            result = await self.gateway.decide(state, questions, model=model)
            call.update(ok=True, output=result)
            return result
        except BaseException as error:
            call.update(ok=False, error=f"{type(error).__name__}: 판단 호출 실패")
            raise
        finally:
            call["latency_ms"] = (time.perf_counter() - started) * 1000


def profile_snapshot(state, analysis=None):
    if analysis is None:
        analysis = ProfileAnalysis(
            state=state,
            profile=state.profile,
            summary=None,
            taste_keywords=(),
            interest_keywords=(),
            keyword_scores={},
            readiness=recommendation_readiness(state),
        )
    payload = {"profile": recommendation_profile(analysis)}
    signals, guide = recommendation_inputs(payload)
    queries = build_search_queries(signals)
    return {
        "profile": payload["profile"],
        "queries": [
            {
                "id": q.query_id,
                "text": q.text,
                "space": q.space.value,
                "modes": sorted(mode.value for mode in q.modes),
                "source_field": q.signal.field.value,
                "source_value": q.signal.value,
            }
            for q in queries
        ],
        "exclusions": list(guide.exclusions),
        "preferences": list(guide.preferences),
        "embedding_inputs": [
            *[{"kind": "search", "text": q.text, "space": q.space.value} for q in queries],
            *[{"kind": "exclusion", "text": text, "space": "content"} for text in guide.exclusions],
            *[
                {"kind": "preference", "text": text, "space": "content"}
                for text in guide.preferences
            ],
        ]
        if queries
        else [],
    }


class RecordingEmbedder:
    def __init__(self, delegate):
        self.delegate = delegate
        self.batches = []

    @property
    def space_id(self):
        return self.delegate.space_id

    async def embed_queries(self, texts):
        started = time.perf_counter()
        batch = {"texts": list(texts)}
        self.batches.append(batch)
        try:
            vectors = await self.delegate.embed_queries(texts)
            batch.update(
                ok=True,
                count=len(vectors),
                vectors=[
                    {"dimensions": len(v), "norm": sqrt(sum(x * x for x in v)), "preview": v[:8]}
                    for v in vectors
                ],
            )
            return vectors
        except Exception as error:
            batch.update(ok=False, error=f"{type(error).__name__}: 임베딩 호출 실패")
            raise
        finally:
            batch["latency_ms"] = (time.perf_counter() - started) * 1000


async def finish_recommendation(snapshot, config, settings, client, analysis, catalog=None):
    stage = config["recommendation_stage"]
    snapshot["recommendation_stage"] = stage
    if stage == "queries":
        snapshot["embedding_status"] = "not_requested"
        snapshot["recommendation_status"] = "not_requested"
        return
    if not snapshot["queries"]:
        snapshot["embedding_status"] = "no_queries"
        snapshot["recommendation_status"] = "no_queries"
        return
    embedding = config["embedding"]
    recorder = RecordingEmbedder(
        OpenAICompatibleEmbedder(
            client=client,
            base_url=embedding["base_url"],
            space_id=embedding["space_id"],
            passage_model=settings.embedding_passage_model_name,
            query_model=embedding["model"],
            dimensions=embedding["dimensions"],
            batch_size=embedding["batch_size"],
            timeout_seconds=embedding["timeout_seconds"],
            api_key=settings.upstage_api_key.get_secret_value()
            if settings.upstage_api_key
            else None,
        )
    )
    pool = None
    try:
        if stage == "embedding":
            await recorder.embed_queries([item["text"] for item in snapshot["embedding_inputs"]])
            snapshot["recommendation_status"] = "not_requested"
        else:
            if catalog is None:
                import asyncpg

                from app.infrastructure.persistence.postgres_catalog_repository import (
                    PostgresCatalogRepository,
                )

                pool = await asyncpg.create_pool(
                    dsn=settings.catalog_database_url_ro or settings.catalog_database_url,
                    min_size=1,
                    max_size=settings.catalog_pool_max_size,
                    server_settings={"default_transaction_read_only": "on"},
                )
                catalog = PostgresCatalogRepository(pool=pool)
            service = V1RecommendationService(
                RecommendationEngine(embedder=recorder, catalog=catalog)
            )
            snapshot["recommendations"] = await service.recommend_lists(
                {"profile": recommendation_profile(analysis)}
            )
            snapshot["recommendation_status"] = "completed"
        snapshot["embedding_status"] = "completed"
    finally:
        snapshot["embedding_batches"] = recorder.batches
        if pool:
            await pool.close()


def build_chat(config, settings, client, calls, session_backend=None):
    gateways = {
        role: RecordingGateway(client, config[role], settings, calls, role)
        for role in ("response", "extraction")
    }
    gateway = RoleGateway(gateways)
    judgment = (
        RecordingJudgment(client, config["judgment"], settings, calls)
        if config["judgment_enabled"]
        else None
    )
    conversations = ConversationService(
        model_gateway=gateway,
        extraction_model="extraction",
        extraction_timeout_seconds=config["extraction"]["timeout_seconds"],
        judgment_gateway=judgment,
        judgment_model=config["judgment"]["model"],
        judgment_timeout_seconds=config["judgment"]["timeout_seconds"],
    )
    states = ConversationStateStore(
        session_backend or InMemorySessionStore(),
        idle_ttl_seconds=config["session_idle_ttl_seconds"],
        max_lifetime_seconds=config["session_max_lifetime_seconds"],
    )
    chat = ChatUseCases(
        conversations=conversations,
        states=states,
        model_gateway=gateway,
        response_model="response",
        response_timeout_seconds=config["response"]["timeout_seconds"],
        summary_timeout_seconds=config["summary_timeout_seconds"],
    )
    return chat, states


async def execute_needu(store, run, stop=None, *, settings=None, client=None, catalog=None):
    settings = settings or load_service_settings()
    config = run["manifest"]["pipeline"]
    if config["environment_hash"] != environment_hash(settings):
        store.finish(run["id"], "failed", {}, "환경변수 설정이 변경됐습니다. 새 실험을 생성하세요.")
        return
    own_client = client is None
    for role in ("response", "extraction"):
        model_key(config[role], settings)
    client = client or httpx.AsyncClient(timeout=None, follow_redirects=False)
    calls = []
    chat, _ = build_chat(config, settings, client, calls)
    utterances = [line.strip() for line in run["spec"]["prompt"].splitlines() if line.strip()]
    simulated = run["spec"].get("dialogue_mode", "fixed") == "simulated"
    planned_turns = run["spec"]["max_turns"] if simulated else len(utterances)
    simulator = (
        RecordingGateway(client, config["persona"], settings, calls, "persona")
        if simulated
        else None
    )
    samples = []
    started = time.perf_counter()
    current = None
    current_started = started
    call_start = 0
    session = None
    stop_reason = None

    def begin(label, user_input, stage, turn):
        nonlocal current, current_started, call_start
        current = {"label": label, "input": user_input, "stage": stage, "turn": turn}
        current_started = time.perf_counter()
        call_start = len(calls)

    def record(output, snapshot, ok=True, error=None):
        nonlocal current
        if "queries" in snapshot and run["manifest"].get("persona"):
            snapshot["query_review"] = review_queries(
                run["manifest"]["persona"], snapshot["queries"], snapshot.get("exclusions", [])
            )
        sample = jsonable_encoder(
            {
                **current,
                "ok": ok,
                "output": output,
                "pipeline": snapshot,
                "latency_ms": (time.perf_counter() - current_started) * 1000,
                "first_chunk_ms": None,
                "usage": None,
                "error": error,
                "model_calls": calls[call_start:],
            }
        )
        store.sample(run["id"], len(samples), sample)
        samples.append(sample)
        current = None

    async def session_flow():
        nonlocal session, stop_reason
        begin("시작 인사", "", "opening", 0)
        opening = await chat.create_session(user_id=1, conversation_room_id=1)
        session = opening.state
        record(opening.greeting, {})
        for index in range(1, planned_turns + 1):
            if session.status != SessionStatus.ACTIVE:
                stop_reason = f"서비스 종료 정책: {session.status.value}"
                break
            begin(f"대화 {index}턴", "", "turn", index)
            persona_ended = False
            if simulated:
                async with asyncio.timeout(config["persona"]["timeout_seconds"]):
                    generated = await simulator.structured(
                        persona_messages(
                            run["manifest"]["persona_simulator"]["system"], session.history
                        ),
                        model=config["persona"]["model"],
                        json_schema=PersonaReply.model_json_schema(),
                    )
                reply = PersonaReply.model_validate(generated)
                utterance = reply.message.strip()
                if not utterance:
                    raise ValueError("페르소나 모델이 빈 발화를 반환했습니다.")
                persona_ended = reply.end_conversation
            else:
                utterance = utterances[index - 1]
            current["input"] = utterance
            result = await chat.send_message(1, utterance, user_id=1)
            session = result.state
            snapshot = profile_snapshot(session)
            snapshot.update(
                extraction_failed=result.extraction_failed,
                session_status=session.status.value,
                readiness=jsonable_encoder(result.readiness),
                merge_result=jsonable_encoder(result.merge_result),
                next_decision=jsonable_encoder(result.next_decision),
                dialogue_mode="simulated" if simulated else "fixed",
                persona_ended=persona_ended,
            )
            record(
                result.reply,
                snapshot,
                ok=not result.extraction_failed,
                error="서비스가 추출 실패 후 빈 델타로 계속했습니다."
                if result.extraction_failed
                else None,
            )
            if persona_ended:
                stop_reason = "페르소나 사용자 종료"
                break
        begin("프로필 분석 · 추천", "", "analysis", session.turn_count)
        analysis = await chat.analyze_session(1)
        snapshot = profile_snapshot(session, analysis)
        snapshot["summary_generated"] = analysis.summary is not None
        snapshot["stop_reason"] = stop_reason
        try:
            await finish_recommendation(snapshot, config, settings, client, analysis, catalog)
        except Exception:
            # Keep the successful profile/query stages visible if embedding/catalog fails.
            record(
                analysis.summary or "프로필 분석 완료",
                snapshot,
                ok=False,
                error="실제 임베딩 또는 카탈로그 추천 실패. 환경 설정과 모델·DB 연결을 확인하세요.",
            )
            raise
        record(analysis.summary or "프로필 분석 완료 · 요약 미생성", snapshot)

    task = asyncio.create_task(session_flow())
    status = "completed"
    try:
        while not task.done():
            if store.cancelled(run["id"]) or (stop and stop.is_set()):
                status = "cancelled"
                break
            if time.perf_counter() - started >= run["spec"]["duration_seconds"]:
                status = "timed_out"
                break
            await asyncio.sleep(0.15)
        if status == "completed":
            await task
    except Exception as error:
        status = "failed"
        if current:
            snapshot = profile_snapshot(session) if session else {}
            record(
                "", snapshot, ok=False, error=f"{type(error).__name__}: 니쥬 파이프라인 실행 실패"
            )
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if current:
            snapshot = profile_snapshot(session) if session else {}
            record("", snapshot, ok=False, error="실행 취소 또는 시간 상한 도달")
        if own_client:
            await client.aclose()
        summary = summarize(samples, time.perf_counter() - started, run["spec"]["requests"])
        turn_samples = [s for s in samples if s["stage"] == "turn"]
        summary.update(
            query_review=next(
                (
                    s["pipeline"]["query_review"]
                    for s in reversed(samples)
                    if "query_review" in s["pipeline"]
                ),
                None,
            ),
            planned_turns=planned_turns,
            dialogue_mode="simulated" if simulated else "fixed",
            executed_turns=len(turn_samples),
            stop_reason=stop_reason,
            extraction_failed_turns=sum(
                bool(s["pipeline"].get("extraction_failed")) for s in turn_samples
            ),
            model_calls=len(calls),
            turn_latency_p50_ms=percentile([s["latency_ms"] for s in turn_samples if s["ok"]], 0.5),
            turn_latency_p95_ms=percentile(
                [s["latency_ms"] for s in turn_samples if s["ok"]], 0.95
            ),
            role_latency_p95_ms={
                role: percentile(
                    [
                        c["latency_ms"]
                        for sample in turn_samples
                        for c in sample["model_calls"]
                        if c["role"] == role and c.get("ok")
                    ],
                    0.95,
                )
                for role in ("response", "extraction", "judgment", "persona")
            },
            model_call_failures=sum(not c.get("ok", False) for c in calls),
            pipeline_latency_ms={
                role: sum(c.get("latency_ms", 0) for c in calls if c["role"] == role)
                for role in ("response", "extraction", "judgment", "persona")
            },
        )
        store.finish(run["id"], status, summary)
    return summary
