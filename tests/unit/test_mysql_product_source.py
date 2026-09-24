from app.infrastructure.persistence.mysql_product_source import _PAGE_SQL


def test_product_source_maps_backend_platform_type_column() -> None:
    assert "platform_type AS platform" in _PAGE_SQL
    assert "WHERE (platform_type, external_id)" in _PAGE_SQL
    assert "ORDER BY platform_type, external_id" in _PAGE_SQL
