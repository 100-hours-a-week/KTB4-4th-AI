from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from app.application.catalog_document_service import CatalogDocumentService
from app.application.catalog_enrichment_service import CatalogEnrichmentService, EnrichmentReport
from app.core.config import Settings, get_settings
from app.infrastructure.embedding import OpenAICompatibleEmbedder
from app.infrastructure.model_gateway import OpenAICompatibleModelGateway

logger = logging.getLogger(__name__)

# 같은 배치가 겹쳐 돌면 LLM 호출이 두 배로 나간다.
# 프로세스가 죽으면 연결이 끊기며 락도 자동으로 풀린다.
_ADVISORY_LOCK_KEY = 883_141_001


class MissingConnectionSettings(RuntimeError):
    """실행에 필요한 설정이 비어 있을 때."""


class AlreadyRunning(RuntimeError):
    """다른 문서 생성 배치가 이미 돌고 있을 때."""


@asynccontextmanager
async def build_service(
    settings: Settings,
) -> AsyncIterator[tuple[CatalogEnrichmentService, Any]]:
    import asyncpg

    from app.infrastructure.persistence.postgres_catalog_document_repository import (
        PostgresCatalogDocumentRepository,
    )

    missing = [
        name
        for name, value in (
            ("CATALOG_DATABASE_URL", settings.catalog_database_url),
            ("DOCUMENT_MODEL_BASE_URL", settings.document_model_base_url),
            ("UPSTAGE_API_KEY", settings.upstage_api_key),
        )
        if not value
    ]
    if missing:
        raise MissingConnectionSettings(f"missing settings: {', '.join(missing)}")

    http_client = httpx.AsyncClient(timeout=settings.document_model_timeout_seconds)
    pool = await asyncpg.create_pool(
        dsn=settings.catalog_database_url,
        min_size=settings.catalog_pool_min_size,
        max_size=settings.catalog_pool_max_size,
    )
    try:
        gateway = OpenAICompatibleModelGateway(
            client=http_client,
            model_base_urls={
                settings.document_model_name: str(settings.document_model_base_url),
            },
            model_max_tokens={
                settings.document_model_name: settings.document_model_max_tokens,
            },
            model_enable_thinking=(
                {
                    settings.document_model_name: settings.document_model_enable_thinking,
                }
                if settings.document_model_enable_thinking is not None
                else None
            ),
            model_temperatures=(
                {
                    settings.document_model_name: settings.document_model_temperature,
                }
                if settings.document_model_temperature is not None
                else None
            ),
            model_stop_sequences=(
                {
                    settings.document_model_name: (settings.document_model_stop_sequence,),
                }
                if settings.document_model_stop_sequence is not None
                else None
            ),
            api_key=(
                settings.model_api_key.get_secret_value()
                if settings.model_api_key is not None
                else None
            ),
        )
        embedder = OpenAICompatibleEmbedder(
            client=http_client,
            base_url=str(settings.embedding_base_url),
            space_id=settings.embedding_space_id,
            passage_model=settings.embedding_passage_model_name,
            query_model=settings.embedding_query_model_name,
            dimensions=settings.embedding_dimensions,
            api_key=(
                settings.upstage_api_key.get_secret_value()
                if settings.upstage_api_key is not None
                else None
            ),
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        service = CatalogEnrichmentService(
            documents=CatalogDocumentService(
                model_gateway=gateway,
                model=settings.document_model_name,
                concurrency=settings.document_model_concurrency,
            ),
            embedder=embedder,
            repository=PostgresCatalogDocumentRepository(pool=pool),
            batch_size=settings.catalog_enrich_batch_size,
            stale_processing_seconds=settings.catalog_enrich_stale_seconds,
        )
        yield service, pool
    finally:
        await pool.close()
        await http_client.aclose()


@asynccontextmanager
async def _exclusive(pool: Any) -> AsyncIterator[None]:
    connection = await pool.acquire()
    try:
        if not await connection.fetchval("SELECT pg_try_advisory_lock($1)", _ADVISORY_LOCK_KEY):
            raise AlreadyRunning("another enrichment batch holds the lock")
        try:
            yield
        finally:
            await connection.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
    finally:
        await pool.release(connection)


async def run_enrichment(
    *,
    max_batches: int = 1,
    retry_failed: bool = False,
) -> EnrichmentReport:
    settings = get_settings()
    async with build_service(settings) as (service, pool):
        async with _exclusive(pool):
            report = await service.run_until_drained(
                max_batches=max_batches,
                retry_failed=retry_failed,
            )
    logger.info(
        "catalog enrichment finished",
        extra={
            "claimed": report.claimed,
            "succeeded": report.succeeded,
            "failed": report.failed,
            "released": report.released,
            "requeued": report.requeued,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate content/usage/gift documents for pending catalog products",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="한 실행에서 처리할 배치 수. 배포 주기보다 짧게 끊는 것이 좋다.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="실패한 활성 상품을 pending 으로 되돌린 뒤 다시 처리한다.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level)
    try:
        report = asyncio.run(
            run_enrichment(
                max_batches=args.max_batches,
                retry_failed=args.retry_failed,
            )
        )
    except AlreadyRunning:
        print("another enrichment batch is running; exiting")
        return
    print(
        f"claimed={report.claimed} succeeded={report.succeeded} "
        f"failed={report.failed} released={report.released}"
        f" requeued={report.requeued}"
    )


if __name__ == "__main__":
    main()
