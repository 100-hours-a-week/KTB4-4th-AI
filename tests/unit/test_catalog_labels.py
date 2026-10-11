from __future__ import annotations

from decimal import Decimal

from app.domain.catalog import (
    CategorySource,
    GiftTheme,
    ProductCategory,
    is_gift_available,
    split_labels,
)
from app.domain.catalog.sync import SourceProduct
from app.domain.recommendation import ProductKey


def test_split_labels_separates_category_and_themes() -> None:
    labels = split_labels(["식품", "감사", "이사·집들이", "친구", "모름"])

    assert labels.category is ProductCategory.FOOD
    assert labels.themes == (GiftTheme.HOUSEWARMING, GiftTheme.THANKS, GiftTheme.FRIEND)
    assert labels.unknown == ("모름",)


def test_liquor_is_not_a_catalog_category() -> None:
    # 니쥬는 전체 이용가라 주류 카테고리가 없다.
    assert split_labels(["주류"]).category is None


def test_gift_available_is_themed_or_voucher_or_luxury() -> None:
    assert is_gift_available(ProductCategory.BEAUTY, [GiftTheme.BIRTHDAY])
    assert is_gift_available(ProductCategory.VOUCHER, [])
    assert is_gift_available(ProductCategory.LUXURY, [])
    assert not is_gift_available(ProductCategory.BEAUTY, [])
    assert not is_gift_available(None, [])


def _source(labels: tuple[str, ...], name: str = "호텔 수건 5P") -> SourceProduct:
    return SourceProduct(
        key=ProductKey(platform="shop", external_id="1"),
        name=name,
        price=Decimal(30000),
        labels=labels,
    )


def test_source_product_uses_source_category_and_themes() -> None:
    product = _source(("리빙", "이사/집들이", "친구"))

    assert product.category_assignment == (ProductCategory.LIVING, CategorySource.SOURCE)
    assert product.themes == (GiftTheme.HOUSEWARMING, GiftTheme.FRIEND)
    assert product.gift_available is True


def test_source_product_without_category_is_left_for_the_model() -> None:
    product = _source(("이사/집들이",))

    assert product.category_assignment == (None, None)
    assert product.to_document_input().category is None


def test_doc_input_hash_tracks_category_and_themes() -> None:
    base = _source(("리빙",))
    themed = _source(("리빙", "친구"))
    renamed = _source(("리빙",), name="호텔 수건 3P")

    assert base.doc_input_hash != themed.doc_input_hash
    assert base.doc_input_hash != renamed.doc_input_hash
    assert base.doc_input_hash == _source(("리빙",)).doc_input_hash
