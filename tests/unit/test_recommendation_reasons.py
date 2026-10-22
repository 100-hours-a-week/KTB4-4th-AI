import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal

from app.application.recommendation_reason_service import RecommendationReasonService
from app.domain.profile.models import DeferralReason, TasteField, Visibility
from app.domain.recommendation import (
    CatalogProduct,
    ProductKey,
    RecommendationMode,
    RecommendationSignal,
    SearchQuery,
    VectorMatch,
    VectorSpace,
    rank_recommendations,
)
from app.domain.recommendation.reasons import reason_clues

NOW = datetime(2026, 10, 22, tzinfo=UTC)


def _query(
    query_id: str,
    field: TasteField,
    value: str,
    *,
    visibility: Visibility = Visibility.FRIENDS,
    deferral_reason: DeferralReason | None = None,
) -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        signal=RecommendationSignal(
            field=field,
            value=value,
            confidence=0.9,
            visibility=visibility,
            updated_at=NOW,
            deferral_reason=deferral_reason,
        ),
        modes=frozenset({RecommendationMode.SELF, RecommendationMode.GIFT}),
        space=VectorSpace.CONTENT,
        text=value,
    )


def _product(external_id: str, name: str) -> CatalogProduct:
    return CatalogProduct(
        key=ProductKey(platform="coupang", external_id=external_id),
        name=name,
        price=Decimal("20000.00"),
        gift_available=True,
    )


def _ranked(mode: RecommendationMode, *pairs: tuple[SearchQuery, CatalogProduct]):
    return rank_recommendations(
        [VectorMatch(query=query, product=product, similarity=0.8) for query, product in pairs],
        mode=mode,
        now=NOW,
    )


class FakeGateway:
    def __init__(self, answers) -> None:
        self.answers = list(answers)
        self.requests: list[dict] = []

    async def structured(self, messages, *, model, json_schema):
        self.requests.append(json.loads(messages[1]["content"]))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_gift_clues_hide_private_values_and_mark_hesitant() -> None:
    hidden = _query(
        "u",
        TasteField.UNAFFORDABLE,
        "고급 그라인더",
        visibility=Visibility.PRIVATE,
        deferral_reason=DeferralReason.PRICE,
    )
    shown = _query("h", TasteField.HOBBIES, "핸드드립")
    (item,) = _ranked(
        RecommendationMode.GIFT, (hidden, _product("1", "핸드밀")), (shown, _product("1", "핸드밀"))
    )

    clues = reason_clues(item.matches, item.mode)

    assert [like.value for like in clues.likes] == ["핸드드립"]
    assert clues.hidden == ("고급 그라인더",)
    assert clues.hesitant


def test_rewritten_reasons_replace_templates_and_bad_ones_are_dropped() -> None:
    lotion = _query("p", TasteField.HOBBIES, "샤워 후 바디로션 바르기")
    wants = _query("w", TasteField.WANTS, "고급 바디오일", visibility=Visibility.PRIVATE)
    ranked = _ranked(
        RecommendationMode.GIFT,
        (lotion, _product("1", "바디로션 세트")),
        (wants, _product("2", "바디오일")),
        (lotion, _product("3", "샤워 가운")),
    )
    gateway = FakeGateway(
        [
            {
                "reasons": [
                    {
                        "id": 0,
                        "reason": "샤워하고 바디케어를 꼭 챙기는 가온님에게 딱 맞는 선물이에요.",
                    },
                    {"id": 1, "reason": "고급 바디오일을 갖고 싶어 한 가온님에게 좋아요."},
                    {"id": 2, "reason": "짧음"},
                ]
            }
        ]
    )
    service = RecommendationReasonService(model_gateway=gateway, model="extractor")

    rewritten = asyncio.run(service.rewrite(ranked, recipient_name="가온"))

    by_name = {item.product.name: item.product.key for item in ranked}
    assert rewritten == {
        by_name["바디로션 세트"]: "샤워하고 바디케어를 꼭 챙기는 가온님에게 딱 맞는 선물이에요."
    }
    sent = {item["product"]: item for item in gateway.requests[0]["items"]}
    assert sent["바디오일"]["likes"] == []
    assert sent["바디로션 세트"]["likes"] == [
        {"value": "샤워 후 바디로션 바르기", "kind": "즐겨 하는 것"}
    ]


def test_failed_chunk_keeps_template_reasons_and_lists_are_split_into_chunks() -> None:
    query = _query("i", TasteField.INTERESTS, "캠핑")
    ranked = _ranked(
        RecommendationMode.SELF,
        *((query, _product(str(index), f"캠핑 용품 {index}")) for index in range(7)),
    )
    gateway = FakeGateway(
        [TimeoutError(), {"reasons": [{"id": 0, "reason": "캠핑 갈 때 챙기기 좋은 용품이에요."}]}]
    )
    service = RecommendationReasonService(model_gateway=gateway, model="extractor")

    rewritten = asyncio.run(service.rewrite(ranked))

    assert [len(request["items"]) for request in gateway.requests] == [5, 2]
    assert list(rewritten.values()) == ["캠핑 갈 때 챙기기 좋은 용품이에요."]
