import pytest

from app.infrastructure.persistence.mysql_product_source import (
    _PAGE_SQL,
    MySQLProductSource,
    _to_product,
    parse_category_labels,
)


def test_product_source_maps_backend_platform_type_column() -> None:
    assert "platform_type AS platform" in _PAGE_SQL
    assert "WHERE (platform_type, external_id)" in _PAGE_SQL
    assert "ORDER BY platform_type, external_id" in _PAGE_SQL


def test_category_column_is_selected_only_when_configured() -> None:
    plain = MySQLProductSource(pool=None, table="products")  # type: ignore[arg-type]
    labeled = MySQLProductSource(
        pool=None,  # type: ignore[arg-type]
        table="products",
        category_column="product_category",
    )

    assert "product_category" not in plain._sql
    assert ", product_category AS product_category" in labeled._sql


def test_themes_column_is_selected_when_configured() -> None:
    source = MySQLProductSource(
        pool=None,  # type: ignore[arg-type]
        table="products",
        category_column="category",
        themes_column="gift_themes",
    )

    assert ", category AS product_category, gift_themes AS gift_themes" in source._sql


def test_category_column_name_must_be_an_identifier() -> None:
    with pytest.raises(ValueError):
        MySQLProductSource(pool=None, table="products", category_column="x; DROP")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        "식품|감사|친구",
        '["식품", "감사", "친구"]',
        ["식품", "감사", "친구"],
        b"\xec\x8b\x9d\xed\x92\x88|\xea\xb0\x90\xec\x82\xac|\xec\xb9\x9c\xea\xb5\xac",
    ],
)
def test_parse_category_labels_accepts_text_json_and_bytes(value: object) -> None:
    assert parse_category_labels(value) == ("식품", "감사", "친구")


def test_row_with_categories_becomes_gift_available_product() -> None:
    product = _to_product(
        {
            "platform": "shop",
            "external_id": "101",
            "name": "고디바 골드 컬렉션 9개입",
            "price": 32000,
            "product_category": "식품",
            "gift_themes": '["감사", "친구"]',
        }
    )

    assert product.labels == ("식품", "감사", "친구")
    assert product.gift_available is True
