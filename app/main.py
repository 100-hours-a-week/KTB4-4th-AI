from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from redis.asyncio import Redis

from app.api.router import api_router
from app.application.chat_use_cases import ChatUseCases
from app.application.conversation_service import ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.ports.catalog_repository import CatalogRepository
from app.application.ports.embedder import Embedder
from app.application.ports.model_gateway import ModelGateway
from app.application.ports.recommendation_service import RecommendationService
from app.application.ports.session_store import SessionStore
from app.application.recommendation_engine import RecommendationEngine
from app.application.recommendation_service import V1RecommendationService
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.infrastructure.embedding import OpenAICompatibleEmbedder
from app.infrastructure.model_gateway import OpenAICompatibleModelGateway
from app.infrastructure.persistence import InMemorySessionStore, RedisSessionStore


def create_app(
    settings: Settings | None = None,
    *,
    model_gateway: ModelGateway | None = None,
    session_store: SessionStore | None = None,
    recommendation_service: RecommendationService | None = None,
    embedder: Embedder | None = None,
    catalog_repository: CatalogRepository | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        http_client: httpx.AsyncClient | None = None
        redis_client: Redis | None = None
        catalog_pool: Any | None = None

        resolved_gateway = model_gateway
        if resolved_gateway is None:
            http_client = httpx.AsyncClient(timeout=None)
            model_base_urls = {}
            if resolved_settings.response_model_base_url is not None:
                model_base_urls[resolved_settings.response_model_name] = str(
                    resolved_settings.response_model_base_url
                )
            if resolved_settings.extraction_model_base_url is not None:
                model_base_urls[resolved_settings.extraction_model_name] = str(
                    resolved_settings.extraction_model_base_url
                )
            if resolved_settings.document_model_base_url is not None:
                model_base_urls[resolved_settings.document_model_name] = str(
                    resolved_settings.document_model_base_url
                )
            secret = resolved_settings.model_api_key or resolved_settings.openrouter_api_key
            resolved_gateway = OpenAICompatibleModelGateway(
                client=http_client,
                model_base_urls=model_base_urls,
                model_max_tokens={
                    resolved_settings.response_model_name: (
                        resolved_settings.response_model_max_tokens
                    ),
                    resolved_settings.extraction_model_name: (
                        resolved_settings.extraction_model_max_tokens
                    ),
                    resolved_settings.document_model_name: (
                        resolved_settings.document_model_max_tokens
                    ),
                },
                api_key=secret.get_secret_value() if secret is not None else None,
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

        state_store = ConversationStateStore(
            resolved_session_store,
            idle_ttl_seconds=resolved_settings.chat_session_idle_ttl_seconds,
            max_lifetime_seconds=resolved_settings.chat_session_max_lifetime_seconds,
        )
        conversation_service = ConversationService(
            model_gateway=resolved_gateway,
            extraction_model=resolved_settings.extraction_model_name,
            extraction_timeout_seconds=resolved_settings.extraction_model_timeout_seconds,
        )
        application.state.conversation_state_store = state_store
        application.state.chat_use_cases = ChatUseCases(
            conversations=conversation_service,
            states=state_store,
            model_gateway=resolved_gateway,
            response_model=resolved_settings.response_model_name,
            response_timeout_seconds=resolved_settings.response_model_timeout_seconds,
            summary_timeout_seconds=resolved_settings.summary_model_timeout_seconds,
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
                )
            )
        application.state.recommendation_service = resolved_recommendation_service
        try:
            yield
        finally:
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
