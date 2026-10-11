"""미리 작성한 상품 문서를 카탈로그에 적재한다.

문서 생성 배치(catalog_enrich)는 LLM 으로 문서를 쓰지만, 이 잡은 이미 써 둔 문서를 읽어
임베딩만 돌린다. 상품 파일은 catalog_products 의 최종 값으로 보고 그대로 넣는다.

    uv run python -m app.jobs.catalog_import \\
        --products products_ai.csv --documents products_ai_documents.jsonl

--ids 로 일부 상품만 넣을 수 있다. 로컬에서 추천 결과를 볼 때 쓴다.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from app.application.catalog_document_service import ProductEnrichment
from app.core.config import Settings, get_settings
from app.domain.catalog import (
    CategorySource,
    DocumentEvidence,
    GiftTheme,
    ProductCategory,
    ProductDocument,
    ProductDocumentSet,
    price_band,
)
from app.domain.recommendation.models import ProductKey, VectorSpace
from app.infrastructure.embedding import OpenAICompatibleEmbedder

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENT_VERSION = "imported-claude-v1"
# 한 번 저장할 때 묶는 상품 수. 상품당 문서가 최대 3개라 임베딩 요청 하나(100개)에 들어간다.
_PRODUCTS_PER_BATCH = 30

_FIELDS = {
    VectorSpace.CONTENT: "contentDocument",
    VectorSpace.USAGE: "usageDocument",
    VectorSpace.GIFT: "giftDocument",
}

# 상품 파일이 최종본이므로 동기화 배치와 달리 판단 결과 컬럼까지 그대로 쓴다.
# 문서가 아직 없으니 pending 으로 넣고, 문서 저장이 ready 로 바꾼다.
_UPSERT_PRODUCT_SQL = """
INSERT INTO catalog_products (
    platform, external_id, name, price,
    source_hash, doc_input_hash, source_labels,
    category, category_source, category_confidence, themes, gift_available,
    subcategory, subcategory_confidence, subcategory_second,
    dedup_group, is_generic_voucher,
    is_active, processing_status
)
SELECT
    p.platform, p.external_id, p.name, p.price,
    p.source_hash, p.doc_input_hash, p.source_labels::text[],
    p.category, p.category_source, p.category_confidence, p.themes::text[], p.gift_available,
    p.subcategory, p.subcategory_confidence, p.subcategory_second,
    p.dedup_group, p.is_generic_voucher,
    p.is_active, 'pending'
FROM jsonb_to_recordset($1::jsonb) AS p(
    platform text, external_id text, name text, price numeric,
    source_hash text, doc_input_hash text, source_labels text,
    category text, category_source text, category_confidence real, themes text,
    gift_available boolean, subcategory text, subcategory_confidence real,
    subcategory_second text, dedup_group text, is_generic_voucher boolean, is_active boolean
)
ON CONFLICT (platform, external_id) DO UPDATE SET
    name                   = EXCLUDED.name,
    price                  = EXCLUDED.price,
    source_hash            = EXCLUDED.source_hash,
    doc_input_hash         = EXCLUDED.doc_input_hash,
    source_labels          = EXCLUDED.source_labels,
    category               = EXCLUDED.category,
    category_source        = EXCLUDED.category_source,
    category_confidence    = EXCLUDED.category_confidence,
    themes                 = EXCLUDED.themes,
    gift_available         = EXCLUDED.gift_available,
    subcategory            = EXCLUDED.subcategory,
    subcategory_confidence = EXCLUDED.subcategory_confidence,
    subcategory_second     = EXCLUDED.subcategory_second,
    dedup_group            = EXCLUDED.dedup_group,
    is_generic_voucher     = EXCLUDED.is_generic_voucher,
    is_active              = EXCLUDED.is_active,
    processing_status      = 'pending',
    processing_error       = NULL,
    updated_at             = now()
"""


class MissingConnectionSettings(RuntimeError):
    """실행에 필요한 설정이 비어 있을 때."""


class DocumentFileError(ValueError):
    """문서 파일이 상품 파일과 맞지 않을 때."""


@dataclass(frozen=True, slots=True)
class ImportReport:
    products: int
    documents: int


def read_products(path: Path, ids: frozenset[str] | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if ids is not None:
        rows = [row for row in rows if row["external_id"] in ids]
    return rows


def read_documents(path: Path) -> dict[str, Mapping[str, Any]]:
    documents: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                documents[row["external_id"]] = row
    return documents


def _optional(value: str) -> str | None:
    return value if value != "" else None


def _product_record(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "platform": row["platform"],
        "external_id": row["external_id"],
        "name": row["name"],
        "price": row["price"] or None,
        "source_hash": row["source_hash"],
        "doc_input_hash": row["doc_input_hash"],
        "source_labels": row["source_labels"] or "{}",
        "category": _optional(row["category"]),
        "category_source": _optional(row["category_source"]),
        "category_confidence": _optional(row["category_confidence"]),
        "themes": row["themes"] or "{}",
        "gift_available": row["gift_available"] == "true",
        "subcategory": _optional(row["subcategory"]),
        "subcategory_confidence": _optional(row["subcategory_confidence"]),
        "subcategory_second": _optional(row["subcategory_second"]),
        "dedup_group": _optional(row["dedup_group"]),
        "is_generic_voucher": row["is_generic_voucher"] == "true",
        "is_active": row.get("is_active", "true") != "false",
    }


def _array(value: str) -> tuple[str, ...]:
    return tuple(item.strip('"') for item in value.strip("{}").split(",") if item)


def build_enrichment(
    row: Mapping[str, str],
    document: Mapping[str, Any],
    *,
    document_version: str,
) -> ProductEnrichment:
    key = ProductKey(platform=row["platform"], external_id=row["external_id"])
    gift_available = row["gift_available"] == "true"
    spaces = [VectorSpace.CONTENT, VectorSpace.USAGE]
    if gift_available:
        spaces.append(VectorSpace.GIFT)
    documents = []
    for space in spaces:
        text = document.get(_FIELDS[space])
        if not isinstance(text, str) or not text.strip():
            raise DocumentFileError(f"{key}: {_FIELDS[space]} is missing")
        documents.append(ProductDocument(space=space, text=text))
    price = Decimal(row["price"]) if row["price"] else None
    return ProductEnrichment(
        key=key,
        documents=ProductDocumentSet(
            key=key,
            documents=tuple(documents),
            evidence=DocumentEvidence.NAME_ONLY,
            document_version=document_version,
        ),
        category=ProductCategory(row["category"]),
        category_source=CategorySource(row["category_source"]),
        category_confidence=(
            float(row["category_confidence"]) if row["category_confidence"] else None
        ),
        themes=tuple(GiftTheme(theme) for theme in _array(row["themes"])),
        gift_available=gift_available,
        price_band=price_band(price),
        attributes={},
    )


def _batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def run_import(
    *,
    products_path: Path,
    documents_path: Path,
    ids: frozenset[str] | None = None,
    document_version: str = DEFAULT_DOCUMENT_VERSION,
    settings: Settings | None = None,
) -> ImportReport:
    import asyncpg

    from app.infrastructure.persistence.postgres_catalog_document_repository import (
        PostgresCatalogDocumentRepository,
    )

    settings = settings or get_settings()
    missing = [
        name
        for name, value in (
            ("CATALOG_DATABASE_URL", settings.catalog_database_url),
            ("UPSTAGE_API_KEY", settings.upstage_api_key),
        )
        if not value
    ]
    if missing:
        raise MissingConnectionSettings(f"missing settings: {', '.join(missing)}")

    rows = read_products(products_path, ids)
    documents = read_documents(documents_path)
    absent = [row["external_id"] for row in rows if row["external_id"] not in documents]
    if absent:
        raise DocumentFileError(f"{len(absent)} products have no documents: {absent[:5]}")
    # 임베딩을 부르기 전에 전부 검증해 두면 중간에 멈춰 반쯤 적재되는 일이 없다.
    enrichments = [
        build_enrichment(row, documents[row["external_id"]], document_version=document_version)
        for row in rows
    ]

    http_client = httpx.AsyncClient(timeout=settings.embedding_timeout_seconds)
    pool = await asyncpg.create_pool(
        dsn=settings.catalog_database_url,
        min_size=settings.catalog_pool_min_size,
        max_size=settings.catalog_pool_max_size,
    )
    try:
        embedder = OpenAICompatibleEmbedder(
            client=http_client,
            base_url=str(settings.embedding_base_url),
            space_id=settings.embedding_space_id,
            passage_model=settings.embedding_passage_model_name,
            query_model=settings.embedding_query_model_name,
            dimensions=settings.embedding_dimensions,
            api_key=settings.upstage_api_key.get_secret_value(),  # type: ignore[union-attr]
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        repository = PostgresCatalogDocumentRepository(pool=pool)

        async with pool.acquire() as connection:
            await connection.execute(
                _UPSERT_PRODUCT_SQL,
                json.dumps([_product_record(row) for row in rows], ensure_ascii=False),
            )
        logger.info("catalog products upserted", extra={"count": len(rows)})

        saved = 0
        for batch in _batches(enrichments, _PRODUCTS_PER_BATCH):
            texts = [text for enrichment in batch for text in enrichment.documents.ordered_texts]
            vectors = await embedder.embed_passages(texts)
            if len(vectors) != len(texts):
                raise ValueError("embedder returned an unexpected number of passage vectors")
            offset = 0
            for enrichment in batch:
                count = len(enrichment.documents.ordered_texts)
                await repository.save_documents(
                    enrichment,
                    vectors=vectors[offset : offset + count],
                    embedding_model_version=embedder.space_id,
                )
                offset += count
            saved += len(texts)
            logger.info("documents saved", extra={"documents": saved, "total_products": len(rows)})
        return ImportReport(products=len(rows), documents=saved)
    finally:
        await pool.close()
        await http_client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import pre-written catalog documents")
    parser.add_argument("--products", type=Path, required=True, help="catalog_products CSV")
    parser.add_argument("--documents", type=Path, required=True, help="documents JSONL")
    parser.add_argument("--ids", type=Path, help="적재할 external_id 목록 파일(한 줄에 하나)")
    parser.add_argument("--document-version", default=DEFAULT_DOCUMENT_VERSION)
    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level)
    ids = None
    if args.ids is not None:
        ids = frozenset(line.strip() for line in args.ids.read_text().splitlines() if line.strip())
    report = asyncio.run(
        run_import(
            products_path=args.products,
            documents_path=args.documents,
            ids=ids,
            document_version=args.document_version,
        )
    )
    print(f"products={report.products} documents={report.documents}")


if __name__ == "__main__":
    main()
