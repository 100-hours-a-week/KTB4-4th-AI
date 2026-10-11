from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import pytest

from app.application.catalog_document_prompts import build_document_messages, build_json_schema
from app.application.catalog_document_service import CatalogDocumentService
from app.domain.catalog import (
    CategorySource,
    DocumentEvidence,
    DocumentSourceProduct,
    DocumentValidationError,
    GiftTheme,
    PriceBand,
    ProductCategory,
    ProductDocument,
    ProductDocumentSet,
    price_band,
)
from app.domain.recommendation import ProductKey, VectorSpace

KEY = ProductKey(platform="shop", external_id="88213")


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contentDocument": "레저·스포츠 캠핑 용품. 티타늄 소재의 350ml 더블월 머그컵.",
        "usageDocument": "캠핑이나 등산에서 따뜻한 음료를 마실 때 쓰는 개인용 머그컵.",
        "giftDocument": "캠핑과 야외 커피를 좋아하는 사람이 받으면 반가운 휴대용 머그컵.",
        "attributes": {"isConsumable": False, "materials": ["티타늄"]},
    }
    payload.update(overrides)
    return payload


def _product(
    name: str = "티타늄 더블월 머그컵 350ml",
    *,
    price: Decimal | None = Decimal(48000),
    category: ProductCategory | None = ProductCategory.SPORTS,
    themes: tuple[GiftTheme, ...] = (GiftTheme.FRIEND,),
    subcategory: str | None = None,
) -> DocumentSourceProduct:
    return DocumentSourceProduct(
        key=KEY,
        name=name,
        price=price,
        category=category,
        category_source=CategorySource.SOURCE if category is not None else None,
        themes=themes,
        subcategory=subcategory,
    )


class StubGateway:
    def __init__(self, payloads: list[Mapping[str, Any] | Exception]) -> None:
        self._payloads = payloads
        self.calls: list[Sequence[Mapping[str, str]]] = []
        self.schemas: list[Mapping[str, Any]] = []

    async def structured(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append(messages)
        self.schemas.append(json_schema)
        result = self._payloads.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _service(gateway: StubGateway) -> CatalogDocumentService:
    return CatalogDocumentService(model_gateway=gateway, model="local-doc-model")


def _user_payload(product: DocumentSourceProduct) -> dict[str, Any]:
    return json.loads(build_document_messages(product)[-1]["content"])


def test_gift_available_product_gets_three_documents() -> None:
    enrichment = asyncio.run(_service(StubGateway([_payload()])).generate(_product()))

    assert enrichment.key == KEY
    assert enrichment.category is ProductCategory.SPORTS
    assert enrichment.category_source is CategorySource.SOURCE
    assert enrichment.category_confidence is None
    assert enrichment.gift_available is True
    assert enrichment.price_band is PriceBand.MID
    assert enrichment.documents.evidence is DocumentEvidence.NAME_ONLY
    assert set(enrichment.documents.spaces) == set(VectorSpace)
    assert "캠핑" in enrichment.documents.text_for(VectorSpace.USAGE)
    assert enrichment.attributes["materials"] == ["티타늄"]


def test_product_outside_gift_available_gets_no_gift_document() -> None:
    gateway = StubGateway([_payload()])
    product = _product(themes=())

    enrichment = asyncio.run(_service(gateway).generate(product))

    # 모델이 gift 문서를 보내도 선물 풀 밖이면 버린다.
    assert enrichment.gift_available is False
    assert enrichment.documents.spaces == (VectorSpace.CONTENT, VectorSpace.USAGE)
    assert "giftDocument" not in gateway.schemas[0]["properties"]


def test_voucher_and_luxury_are_gift_available_without_themes() -> None:
    for category in (ProductCategory.VOUCHER, ProductCategory.LUXURY):
        assert _product(category=category, themes=()).gift_available is True
    assert _product(category=ProductCategory.BEAUTY, themes=()).gift_available is False


def test_gift_available_product_must_return_gift_document() -> None:
    payload = _payload()
    del payload["giftDocument"]

    with pytest.raises(DocumentValidationError):
        asyncio.run(_service(StubGateway([payload])).generate(_product()))


def test_unknown_category_is_inferred_by_the_model() -> None:
    gateway = StubGateway(
        [
            _payload(
                category="리빙",
                categoryConfidence=0.82,
                contentDocument="호텔식 면 수건 세트. 욕실에 두는 타월.",
                usageDocument="샤워 후나 세안 뒤 욕실에서 매일 쓰는 수건.",
                giftDocument="깔끔한 생활용품을 좋아하는 사람이 받으면 반가운 수건 세트.",
            )
        ]
    )
    product = _product(name="오셀픽 호텔 수건", category=None, themes=(GiftTheme.HOUSEWARMING,))

    enrichment = asyncio.run(_service(gateway).generate(product))

    assert enrichment.category is ProductCategory.LIVING
    assert enrichment.category_source is CategorySource.INFERRED
    assert enrichment.category_confidence == pytest.approx(0.82)
    assert gateway.schemas[0]["properties"]["category"]["enum"][0] == "교환권"


def test_known_category_is_not_requested_or_overridden() -> None:
    gateway = StubGateway([_payload(category="식품", categoryConfidence=0.9)])

    enrichment = asyncio.run(_service(gateway).generate(_product()))

    assert enrichment.category is ProductCategory.SPORTS
    assert "category" not in gateway.schemas[0]["properties"]


def test_inferred_category_outside_the_eleven_is_rejected() -> None:
    gateway = StubGateway([_payload(category="hobby", categoryConfidence=0.9)])

    with pytest.raises(DocumentValidationError):
        asyncio.run(_service(gateway).generate(_product(category=None)))


@pytest.mark.parametrize(
    "gift",
    [
        "캠핑을 좋아하는 남자친구에게 건네기 좋은 머그컵.",
        "남자에게 선물하기 좋은 캠핑용 머그컵이에요.",
        "부모님께 드리기 좋은 아웃도어용 머그컵.",
    ],
)
def test_gift_document_cannot_describe_recipient_by_relation_or_gender(gift: str) -> None:
    with pytest.raises(DocumentValidationError):
        asyncio.run(_service(StubGateway([_payload(giftDocument=gift)])).generate(_product()))


def test_generate_rejects_identical_documents() -> None:
    same = "티타늄 소재의 350ml 더블월 머그컵과 아웃도어용 보온 컵."
    gateway = StubGateway([_payload(contentDocument=same, usageDocument=same, giftDocument=same)])

    with pytest.raises(DocumentValidationError):
        asyncio.run(_service(gateway).generate(_product()))


def test_generate_rejects_document_with_price() -> None:
    gateway = StubGateway([_payload(giftDocument="48,000원에 선물하기 좋은 캠핑용 머그컵이에요.")])

    with pytest.raises(DocumentValidationError):
        asyncio.run(_service(gateway).generate(_product()))


def test_generate_rejects_numeric_fact_absent_from_product_name() -> None:
    gateway = StubGateway([_payload(contentDocument="100ml 용량의 라벤더 향 핸드크림 세트입니다.")])

    with pytest.raises(DocumentValidationError):
        asyncio.run(_service(gateway).generate(_product(name="라벤더 향 핸드크림 선물세트")))


def test_generate_discards_material_not_present_in_product_name() -> None:
    gateway = StubGateway([_payload(attributes={"isConsumable": True, "materials": ["핸드크림"]})])
    product = _product(name="라벤더 향 핸드크림 선물세트 350ml", category=ProductCategory.BEAUTY)

    enrichment = asyncio.run(_service(gateway).generate(product))

    assert enrichment.attributes["materials"] == []


def test_generate_does_not_match_one_character_material_substrings() -> None:
    gateway = StubGateway([_payload(attributes={"materials": ["금", "은", "면", "울"]})])

    enrichment = asyncio.run(
        _service(gateway).generate(_product(name="지금 쓰는 서울 면도기 350ml"))
    )

    assert enrichment.attributes["materials"] == []


def test_prompt_uses_name_category_label_and_price_band() -> None:
    product = _product(name="티타늄 더블월 머그컵 350ml", price=Decimal(120000))

    payload = _user_payload(product)

    assert payload["productName"] == "티타늄 더블월 머그컵 350ml"
    assert payload["category"] == "레저·스포츠"
    assert payload["priceBand"] == "high"
    assert "120000" not in json.dumps(payload)


def test_prompt_passes_subcategory_as_korean_label() -> None:
    assert _user_payload(_product(subcategory="camping"))["subcategory"] == "캠핑 용품"
    assert _user_payload(_product())["subcategory"] is None


def test_subcategory_must_belong_to_category() -> None:
    with pytest.raises(ValueError):
        _product(category=ProductCategory.SPORTS, subcategory="skincare")
    with pytest.raises(ValueError):
        _product(category=None, subcategory="camping")


def test_prompt_passes_only_recipient_side_theme_hints() -> None:
    product = _product(
        themes=(GiftTheme.BIRTHDAY, GiftTheme.HOUSEWARMING, GiftTheme.CHEER, GiftTheme.FUN)
    )

    payload = _user_payload(product)

    # 생일·친구 테마는 gift_available 근거로만 쓰고 문서에는 넘기지 않는다.
    assert payload["writeGift"] is True
    assert payload["giftHints"] == ["housewarming", "cheer", "fun"]


def test_prompt_marks_products_outside_gift_available() -> None:
    payload = _user_payload(_product(themes=()))

    assert payload["writeGift"] is False
    assert payload["giftHints"] == []


def test_schema_requests_category_only_when_unknown() -> None:
    known = build_json_schema(_product())
    unknown = build_json_schema(_product(category=None))

    assert "category" not in known["required"]
    assert {"category", "categoryConfidence"} <= set(unknown["required"])


def test_document_set_requires_content_and_usage() -> None:
    with pytest.raises(DocumentValidationError):
        ProductDocumentSet(
            key=KEY,
            documents=(ProductDocument(space=VectorSpace.CONTENT, text="티타늄 머그컵입니다."),),
            evidence=DocumentEvidence.NAME_ONLY,
        )


def test_document_set_rejects_duplicate_space() -> None:
    with pytest.raises(DocumentValidationError):
        ProductDocumentSet(
            key=KEY,
            documents=(
                ProductDocument(space=VectorSpace.CONTENT, text="티타늄 머그컵입니다."),
                ProductDocument(space=VectorSpace.USAGE, text="캠핑에서 쓰는 머그컵."),
                ProductDocument(space=VectorSpace.USAGE, text="등산에서 쓰는 머그컵."),
            ),
            evidence=DocumentEvidence.NAME_ONLY,
        )


def test_source_product_cannot_claim_inferred_category() -> None:
    with pytest.raises(ValueError):
        DocumentSourceProduct(
            key=KEY,
            name="수건",
            category=ProductCategory.LIVING,
            category_source=CategorySource.INFERRED,
        )


def test_price_band_boundaries() -> None:
    assert price_band(None) is None
    assert price_band(Decimal(19999)) is PriceBand.LOW
    assert price_band(Decimal(20000)) is PriceBand.MID
    assert price_band(Decimal(80000)) is PriceBand.HIGH
