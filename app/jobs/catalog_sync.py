from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.application.catalog_sync_service import CatalogSyncService, SyncReport
from app.core.config import Settings, get_settings
from app.domain.catalog.sync import SyncMode

logger = logging.getLogger(__name__)


class MissingConnectionSettings(RuntimeError):
    """DB 접속 정보가 채워지지 않아 동기화를 실행할 수 없을 때."""


@asynccontextmanager
async def build_service(settings: Settings) -> AsyncIterator[CatalogSyncService]:
    """설정에 실제 접속 정보가 채워지면 그대로 연결된다.

    드라이버는 여기서만 import 한다. 동기화를 쓰지 않는 환경에서
    aiomysql / asyncpg 가 없어도 앱이 뜨게 하기 위해서다.
    """
    import aiomysql
    import asyncpg

    from app.infrastructure.persistence.mysql_product_source import MySQLProductSource
    from app.infrastructure.persistence.postgres_catalog_sync_repository import (
        PostgresCatalogSyncRepository,
    )

    missing = [
        name
        for name, value in (
            ("SOURCE_MYSQL_HOST", settings.source_mysql_host),
            ("SOURCE_MYSQL_USER", settings.source_mysql_user),
            ("SOURCE_MYSQL_DATABASE", settings.source_mysql_database),
            ("CATALOG_DATABASE_URL", settings.catalog_database_url),
        )
        if not value
    ]
    if missing:
        raise MissingConnectionSettings(f"missing settings: {', '.join(missing)}")

    mysql_pool = await aiomysql.create_pool(
        host=settings.source_mysql_host,
        port=settings.source_mysql_port,
        user=settings.source_mysql_user,
        password=(
            settings.source_mysql_password.get_secret_value()
            if settings.source_mysql_password is not None
            else ""
        ),
        db=settings.source_mysql_database,
        minsize=1,
        maxsize=settings.source_mysql_pool_size,
        autocommit=True,
        charset="utf8mb4",
    )
    postgres_pool = await asyncpg.create_pool(
        dsn=settings.catalog_database_url,
        min_size=settings.catalog_pool_min_size,
        max_size=settings.catalog_pool_max_size,
    )
    try:
        yield CatalogSyncService(
            source=MySQLProductSource(
                pool=mysql_pool,
                table=settings.source_mysql_table,
            ),
            repository=PostgresCatalogSyncRepository(pool=postgres_pool),
            batch_size=settings.catalog_sync_batch_size,
        )
    finally:
        mysql_pool.close()
        await mysql_pool.wait_closed()
        await postgres_pool.close()


async def run_sync(*, mode: SyncMode = SyncMode.FULL) -> SyncReport:
    settings = get_settings()
    async with build_service(settings) as service:
        report = await service.sync(mode=mode)
    logger.info(
        "catalog sync finished",
        extra={
            "run_id": report.run_id,
            "received": report.received_count,
            "changed": report.changed_count,
            "deactivated": report.deactivated_count,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync backend MySQL products into the AI catalog")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SyncMode],
        default=SyncMode.FULL.value,
    )
    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level)
    report = asyncio.run(run_sync(mode=SyncMode(args.mode)))
    print(
        f"run_id={report.run_id} received={report.received_count} "
        f"changed={report.changed_count} deactivated={report.deactivated_count}"
    )


if __name__ == "__main__":
    main()
