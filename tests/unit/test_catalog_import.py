from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.domain.catalog import (
    CategorySource,
    DocumentValidationError,
    GiftTheme,
    PriceBand,
    ProductCategory,
)
from app.domain.recommendation import ProductKey, VectorSpace
from app.jobs.catalog_import import (
    DocumentFileError,
    _product_record,
    build_enrichment,
    read_documents,
    read_products,
)

_FIELDS = [
    "platform",
    "external_id",
    "name",
    "price",
    "source_labels",
    "category",
    "category_source",
    "category_confidence",
    "themes",
    "gift_available",
    "subcategory",
    "subcategory_confidence",
    "subcategory_second",
    "dedup_group",
    "is_generic_voucher",
    "is_active",
    "source_hash",
    "doc_input_hash",
]


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "platform": "kakao",
        "external_id": "1001",
        "name": "티타늄 더블월 머그컵",
        "price": "32000",
        "source_labels": "{레저·스포츠,집들이}",
        "category": "sports",
        "category_source": "source",
        "category_confidence": "",
        "themes": "{housewarming}",
        "gift_available": "true",
        "subcategory": "camping",
        "subcategory_confidence": "0.97",
        "subcategory_second": "workout:0.02",
        "dedup_group": "a1b2c3d4e5f60718",
        "is_generic_voucher": "false",
        "is_active": "true",
        "source_hash": "s" * 64,
        "doc_input_hash": "d" * 64,
    }
    row.update(overrides)
    return row


def _documents(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "platform": "kakao",
        "external_id": "1001",
        "contentDocument": "레저·스포츠 캠핑 용품. 티타늄 더블월 캠핑 머그컵.",
        "usageDocument": "캠핑장이나 등산길에서 따뜻한 음료를 마실 때 쓰는 머그컵이다.",
        "giftDocument": "캠핑을 좋아하는 사람이 받으면 좋은 캠핑 머그컵.",
    }
    document.update(overrides)
    return document


def test_build_enrichment_keeps_gift_document_for_gift_products() -> None:
    enrichment = build_enrichment(_row(), _documents(), document_version="imported-test")

    assert enrichment.key == ProductKey(platform="kakao", external_id="1001")
    assert enrichment.documents.spaces == (
        VectorSpace.CONTENT,
        VectorSpace.GIFT,
        VectorSpace.USAGE,
    )
    assert enrichment.documents.document_version == "imported-test"
    assert enrichment.category is ProductCategory.SPORTS
    assert enrichment.category_source is CategorySource.SOURCE
    assert enrichment.category_confidence is None
    assert enrichment.themes == (GiftTheme.HOUSEWARMING,)
    assert enrichment.gift_available is True
    assert enrichment.price_band is PriceBand.MID


def test_build_enrichment_ignores_gift_document_for_non_gift_products() -> None:
    enrichment = build_enrichment(
        _row(gift_available="false", themes="{}"),
        _documents(),
        document_version="imported-test",
    )

    assert VectorSpace.GIFT not in enrichment.documents.spaces
    assert enrichment.gift_available is False


def test_build_enrichment_requires_gift_document_for_gift_products() -> None:
    document = _documents()
    del document["giftDocument"]

    with pytest.raises(DocumentFileError, match="giftDocument"):
        build_enrichment(_row(), document, document_version="imported-test")


def test_build_enrichment_rejects_blank_usage_document() -> None:
    with pytest.raises(DocumentFileError, match="usageDocument"):
        build_enrichment(_row(), _documents(usageDocument="  "), document_version="v")


def test_build_enrichment_applies_document_validation() -> None:
    with pytest.raises(DocumentValidationError):
        build_enrichment(
            _row(),
            _documents(contentDocument="가" * 201),
            document_version="imported-test",
        )


def test_product_record_converts_csv_values() -> None:
    record = _product_record(_row(category_confidence="", subcategory_second=""))

    assert record["gift_available"] is True
    assert record["is_generic_voucher"] is False
    assert record["is_active"] is True
    assert record["category_confidence"] is None
    assert record["subcategory_second"] is None
    assert record["source_labels"] == "{레저·스포츠,집들이}"
    assert Decimal(str(record["price"])) == Decimal("32000")


def test_product_record_defaults_empty_arrays() -> None:
    record = _product_record(_row(source_labels="", themes=""))

    assert record["source_labels"] == "{}"
    assert record["themes"] == "{}"


def test_read_products_filters_by_ids(tmp_path: Path) -> None:
    path = tmp_path / "products.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerow(_row(external_id="1001"))
        writer.writerow(_row(external_id="1002"))

    assert [row["external_id"] for row in read_products(path)] == ["1001", "1002"]
    assert [row["external_id"] for row in read_products(path, frozenset({"1002"}))] == ["1002"]


def test_read_documents_indexes_by_external_id(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_documents(external_id=external_id), ensure_ascii=False)
            for external_id in ("1001", "1002")
        )
        + "\n\n",
        encoding="utf-8",
    )

    documents = read_documents(path)

    assert set(documents) == {"1001", "1002"}
    assert documents["1002"]["external_id"] == "1002"
