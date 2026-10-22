from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import SecretStr
from redis.asyncio import Redis

from app.api.router import api_router
from app.application.chat_use_cases import ChatUseCases
from app.application.conversation_service import ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.ports.catalog_repository import CatalogRepository
from app.application.ports.embedder import Embedder
from app.application.ports.judgment_gateway import JudgmentGateway
from app.application.ports.model_gateway import ModelGateway
from app.application.ports.recommendation_service import RecommendationService
from app.application.ports.session_store import SessionStore
from app.application.recommendation_engine import RecommendationEngine
from app.application.recommendation_service import V1RecommendationService
from app.application.search_intent_service import SearchIntentService
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware
from app.infrastructure.embedding import OpenAICompatibleEmbedder
from app.infrastructure.model_gateway import (
    FailoverModelGateway,
    FallbackRoute,
    OpenAICompatibleModelGateway,
    RunPodPodLocator,
    SystemOneJudgmentGateway,
)
from app.infrastructure.persistence import InMemorySessionStore, RedisSessionStore


def _secret_value(secret: SecretStr | None) -> str | None:
    return secret.get_secret_value() if secret is not None else None


def _build_model_gateway(
    settings: Settings,
    client: httpx.AsyncClient,
    *,
    api_key: str | None,
    fallback_api_key: str | None,
    locator: RunPodPodLocator | None = None,
) -> ModelGateway:
    model_base_urls: dict[str, str | Callable[[], str | None]] = {}
    if locator is not None:
        # 응답·추출 1차 주소는 고정값 대신 지금 떠 있는 RunPod Pod에서 정한다.
        model_base_urls[settings.response_model_name] = lambda: locator.base_url
        model_base_urls[settings.extraction_model_name] = lambda: locator.base_url
    else:
        if settings.response_model_base_url is not None:
            model_base_urls[settings.response_model_name] = str(settings.response_model_base_url)
        if settings.extraction_model_base_url is not None:
            model_base_urls[settings.extraction_model_name] = str(
                settings.extraction_model_base_url
            )
    if settings.document_model_base_url is not None:
        model_base_urls[settings.document_model_name] = str(settings.document_model_base_url)

    fallbacks: dict[str, FallbackRoute] = {}
    for primary_model, base_url, fallback_model, max_tokens in (
        (
            settings.response_model_name,
            settings.response_fallback_model_base_url,
            settings.response_fallback_model_name,
            settings.response_model_max_tokens,
        ),
        (
            settings.extraction_model_name,
            settings.extraction_fallback_model_base_url,
            settings.extraction_fallback_model_name,
            settings.extraction_model_max_tokens,
        ),
    ):
        if base_url is None or not fallback_model:
            continue
        fallbacks[primary_model] = FallbackRoute(
            gateway=OpenAICompatibleModelGateway(
                client=client,
                model_base_urls={fallback_model: str(base_url)},
                model_max_tokens={fallback_model: max_tokens},
                api_key=fallback_api_key,
            ),
            model=fallback_model,
        )

    primary = OpenAICompatibleModelGateway(
        client=client,
        model_base_urls=model_base_urls,
        model_max_tokens={
            settings.response_model_name: settings.response_model_max_tokens,
            settings.extraction_model_name: settings.extraction_model_max_tokens,
            settings.document_model_name: settings.document_model_max_tokens,
        },
        api_key=api_key,
        # 폴백이 없으면 기다리는 것 말고 할 수 있는 게 없으니 제한을 걸지 않는다.
        timeout=(
            httpx.Timeout(
                None,
                connect=settings.primary_model_connect_timeout_seconds,
                read=settings.primary_model_read_timeout_seconds,
            )
            if fallbacks
            else None
        ),
    )
    if not fallbacks:
        return primary
    return FailoverModelGateway(
        primary=primary,
        fallbacks=fallbacks,
        cooldown_seconds=settings.primary_model_cooldown_seconds,
        max_cooldown_seconds=settings.primary_model_max_cooldown_seconds,
        primary_ready=locator.is_ready if locator is not None else None,
    )


def _build_pod_locator(settings: Settings, client: httpx.AsyncClient) -> RunPodPodLocator | None:
    api_key = _secret_value(settings.runpod_api_key)
    if not settings.runpod_pod_name_prefix or api_key is None:
        return None
    return RunPodPodLocator(
        client=client,
        api_key=api_key,
        name_prefix=settings.runpod_pod_name_prefix,
        port=settings.runpod_pod_port,
        api_base_url=str(settings.runpod_api_base_url),
        interval_seconds=settings.runpod_lookup_interval_seconds,
    )


def create_app(
    settings: Settings | None = None,
    *,
    model_gateway: ModelGateway | None = None,
    judgment_gateway: JudgmentGateway | None = None,
    session_store: SessionStore | None = None,
    recommendation_service: RecommendationService | None = None,
    embedder: Embedder | None = None,
    catalog_repository: CatalogRepository | None = None,
    search_intent_service: SearchIntentService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        http_client: httpx.AsyncClient | None = None
        redis_client: Redis | None = None
        catalog_pool: Any | None = None

        resolved_gateway = model_gateway
        # 응답·추출 1차(RunPod vLLM)는 MODEL_API_KEY, OpenRouter로 가는 Jev와 폴백은
        # OPENROUTER_API_KEY를 쓴다. 한쪽만 있으면 V1처럼 그 키 하나로 모두 호출한다.
        api_key = _secret_value(
            resolved_settings.model_api_key or resolved_settings.openrouter_api_key
        )
        openrouter_api_key = _secret_value(
            resolved_settings.openrouter_api_key or resolved_settings.model_api_key
        )
        locator_task: asyncio.Task[None] | None = None
        if resolved_gateway is None:
            http_client = httpx.AsyncClient(timeout=None)
            locator = _build_pod_locator(resolved_settings, http_client)
            if locator is not None:
                # 첫 요청부터 맞는 주소로 보내도록 시작할 때 한 번 찾고, 이후는 주기적으로 찾는다.
                await locator.refresh()
                locator_task = asyncio.create_task(locator.run())
            resolved_gateway = _build_model_gateway(
                resolved_settings,
                http_client,
                api_key=api_key,
                fallback_api_key=_secret_value(resolved_settings.openrouter_api_key),
                locator=locator,
            )

        # 모델 게이트웨이를 주입받았으면 판단 게이트웨이도 주입받은 것만 쓴다.
        resolved_judgment_gateway = judgment_gateway
        if (
            resolved_judgment_gateway is None
            and model_gateway is None
            and resolved_settings.judgment_model_base_url is not None
        ):
            if http_client is None:
                http_client = httpx.AsyncClient(timeout=None)
            resolved_judgment_gateway = SystemOneJudgmentGateway(
                client=http_client,
                base_url=str(resolved_settings.judgment_model_base_url),
                api_key=openrouter_api_key,
            )

        resolved_session_store = session_store
        if resolved_session_store is None:
            if resolved_settings.redis_url is None:
                resolved_session_store = InMemorySessionStore()
            else:
                redis_client = Redis.from_url(
                    str(resolved_settings.redis_url),
                    decode_responses=True,
                )
                resolved_session_store = RedisSessionStore(redis_client)

        # 모델 게이트웨이를 주입받은 테스트에서는 주입받은 검색 의도 서비스만 쓴다.
        resolved_intents = search_intent_service
        if (
            resolved_intents is None
            and model_gateway is None
            and resolved_settings.search_intent_enabled
        ):
            from app.infrastructure.persistence.memory_search_intent_store import (
                InMemorySearchIntentStore,
            )
            from app.infrastructure.persistence.redis_search_intent_store import (
                RedisSearchIntentStore,
            )

            resolved_intents = SearchIntentService(
                model_gateway=resolved_gateway,
                model=resolved_settings.extraction_model_name,
                store=(
                    RedisSearchIntentStore(redis_client)
                    if redis_client is not None
                    else InMemorySearchIntentStore()
                ),
                timeout_seconds=resolved_settings.extraction_model_timeout_seconds,
            )

        state_store = ConversationStateStore(
            resolved_session_store,
            idle_ttl_seconds=resolved_settings.chat_session_idle_ttl_seconds,
            max_lifetime_seconds=resolved_settings.chat_session_max_lifetime_seconds,
        )
        conversation_service = ConversationService(
            model_gateway=resolved_gateway,
            extraction_model=resolved_settings.extraction_model_name,
            extraction_timeout_seconds=resolved_settings.extraction_model_timeout_seconds,
            judgment_gateway=resolved_judgment_gateway,
            judgment_model=resolved_settings.judgment_model_name,
            judgment_timeout_seconds=resolved_settings.judgment_model_timeout_seconds,
        )
        application.state.conversation_state_store = state_store
        application.state.chat_use_cases = ChatUseCases(
            conversations=conversation_service,
            states=state_store,
            model_gateway=resolved_gateway,
            response_model=resolved_settings.response_model_name,
            response_timeout_seconds=resolved_settings.response_model_timeout_seconds,
            summary_timeout_seconds=resolved_settings.summary_model_timeout_seconds,
            search_intents=resolved_intents,
            keyword_model=resolved_settings.extraction_model_name,
        )
        resolved_recommendation_service = recommendation_service
        resolved_catalog = catalog_repository
        catalog_read_database_url = (
            resolved_settings.catalog_database_url_ro or resolved_settings.catalog_database_url
        )
        if (
            resolved_catalog is None
            and resolved_recommendation_service is None
            and catalog_read_database_url is not None
        ):
            import asyncpg

            from app.infrastructure.persistence.postgres_catalog_repository import (
                PostgresCatalogRepository,
            )

            catalog_pool = await asyncpg.create_pool(
                dsn=catalog_read_database_url,
                min_size=resolved_settings.catalog_pool_min_size,
                max_size=resolved_settings.catalog_pool_max_size,
            )
            resolved_catalog = PostgresCatalogRepository(pool=catalog_pool)
        if resolved_recommendation_service is None and resolved_catalog is not None:
            resolved_embedder = embedder
            if resolved_embedder is None:
                if http_client is None:
                    http_client = httpx.AsyncClient(timeout=None)
                embedding_secret = resolved_settings.upstage_api_key
                resolved_embedder = OpenAICompatibleEmbedder(
                    client=http_client,
                    base_url=str(resolved_settings.embedding_base_url),
                    space_id=resolved_settings.embedding_space_id,
                    passage_model=resolved_settings.embedding_passage_model_name,
                    query_model=resolved_settings.embedding_query_model_name,
                    dimensions=resolved_settings.embedding_dimensions,
                    api_key=(
                        embedding_secret.get_secret_value()
                        if embedding_secret is not None
                        else None
                    ),
                    batch_size=resolved_settings.embedding_batch_size,
                    timeout_seconds=resolved_settings.embedding_timeout_seconds,
                )
            resolved_recommendation_service = V1RecommendationService(
                RecommendationEngine(
                    embedder=resolved_embedder,
                    catalog=resolved_catalog,
                ),
                intents=resolved_intents,
            )
        application.state.recommendation_service = resolved_recommendation_service
        try:
            yield
        finally:
            if resolved_intents is not None:
                await resolved_intents.aclose()
            if locator_task is not None:
                locator_task.cancel()
                with suppress(asyncio.CancelledError):
                    await locator_task
            if redis_client is not None:
                await redis_client.aclose()
            if http_client is not None:
                await http_client.aclose()
            if catalog_pool is not None:
                await catalog_pool.close()

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.add_middleware(RequestIdMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
