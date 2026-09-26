from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import pytest

from app.application.catalog_document_prompts import build_document_messages
from app.application.catalog_document_service import CatalogDocumentService
from app.domain.catalog import (
    DocumentEvidence,
    DocumentSourceProduct,
    DocumentValidationError,
    PriceBand,
    ProductDocument,
    ProductDocumentSet,
    price_band,
)
from app.domain.recommendation import ProductKey, VectorSpace

KEY = ProductKey(platform="coupang", external_id="88213")


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "categoryCode": "hobby",
        "categoryConfidence": 0.9,
        "contentDocument": "티타늄 소재의 350ml 더블월 머그컵. 보온이 되는 아웃도어용 컵.",
        "usageDocument": "캠핑이나 등산에서 따뜻한 음료를 마실 때 쓰는 개인용 머그컵.",
        "giftDocument": "캠핑과 야외 커피를 즐기는 사람에게 건네기 좋은 휴대용 머그컵.",
        "attributes": {"isConsumable": False, "materials": ["티타늄"]},
    }
    payload.update(overrides)
    return payload


class StubGateway:
    def __init__(self, payloads: list[Mapping[str, Any] | Exception]) -> None:
        self._payloads = payloads
        self.calls: list[Sequence[Mapping[str, str]]] = []

    async def structured(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append(messages)
        result = self._payloads.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _service(payloads: list[Mapping[str, Any] | Exception]) -> CatalogDocumentService:
    return CatalogDocumentService(model_gateway=StubGateway(payloads), model="local-doc-model")


def test_generate_builds_three_documents() -> None:
    service = _service([_payload()])
    product = DocumentSourceProduct(
        key=KEY, name="티타늄 더블월 머그컵 350ml", price=Decimal(48000)
    )

    enrichment = asyncio.run(service.generate(product))

    assert enrichment.key == KEY
    assert enrichment.category_code == "hobby"
    assert enrichment.price_band is PriceBand.MID
    assert enrichment.documents.evidence is DocumentEvidence.NAME_ONLY
    assert len(enrichment.documents.documents) == 3
    assert "캠핑" in enrichment.documents.text_for(VectorSpace.USAGE)
    assert enrichment.attributes["materials"] == ["티타늄"]


def test_generate_rejects_identical_documents() -> None:
    same = "티타늄 소재의 350ml 더블월 머그컵과 아웃도어용 보온 컵."
    service = _service(
        [_payload(contentDocument=same, usageDocument=same, giftDocument=same)],
    )
    product = DocumentSourceProduct(key=KEY, name="티타늄 머그컵")

    with pytest.raises(DocumentValidationError):
        asyncio.run(service.generate(product))


def test_generate_rejects_document_with_price() -> None:
    service = _service([_payload(giftDocument="48,000원에 선물하기 좋은 캠핑용 머그컵이에요.")])
    product = DocumentSourceProduct(key=KEY, name="티타늄 머그컵")

    with pytest.raises(DocumentValidationError):
        asyncio.run(service.generate(product))


def test_generate_rejects_numeric_fact_absent_from_product_name() -> None:
    service = _service(
        [_payload(contentDocument="100ml 용량의 라벤더 향 핸드크림 선물세트입니다.")]
    )
    product = DocumentSourceProduct(key=KEY, name="라벤더 향 핸드크림 선물세트")

    with pytest.raises(DocumentValidationError):
        asyncio.run(service.generate(product))


def test_generate_discards_material_not_present_in_product_name() -> None:
    service = _service([_payload(attributes={"isConsumable": True, "materials": ["핸드크림"]})])
    product = DocumentSourceProduct(
        key=KEY,
        name="라벤더 향 핸드크림 선물세트 350ml",
    )

    enrichment = asyncio.run(service.generate(product))

    assert enrichment.attributes["materials"] == []


def test_generate_does_not_match_one_character_material_substrings() -> None:
    service = _service([_payload(attributes={"materials": ["금", "은", "면", "울"]})])
    product = DocumentSourceProduct(
        key=KEY,
        name="지금 쓰는 서울 면도기 350ml",
    )

    enrichment = asyncio.run(service.generate(product))

    assert enrichment.attributes["materials"] == []


def test_prompt_includes_price_band_not_amount() -> None:
    product = DocumentSourceProduct(key=KEY, name="티타늄 머그컵", price=Decimal(120000))
    messages = build_document_messages(product)

    user_message = messages[-1]["content"]
    assert '"high"' in user_message
    assert "120000" not in user_message


def test_document_set_requires_all_three_spaces() -> None:
    with pytest.raises(DocumentValidationError):
        ProductDocumentSet(
            key=KEY,
            documents=(ProductDocument(space=VectorSpace.CONTENT, text="티타늄 머그컵입니다."),),
            evidence=DocumentEvidence.NAME_ONLY,
        )


def test_price_band_boundaries() -> None:
    assert price_band(None) is None
    assert price_band(Decimal(19999)) is PriceBand.LOW
    assert price_band(Decimal(20000)) is PriceBand.MID
    assert price_band(Decimal(80000)) is PriceBand.HIGH
