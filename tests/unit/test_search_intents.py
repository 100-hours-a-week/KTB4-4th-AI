import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from app.application.chat_use_cases import ChatUseCases
from app.application.conversation_service import ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.recommendation_engine import RecommendationBatch
from app.application.recommendation_service import V1RecommendationService, recommendation_inputs
from app.application.search_intent_prompts import SEARCH_INTENT_SCHEMA
from app.application.search_intent_service import (
    IntentTarget,
    SearchIntentService,
    intent_targets,
    onboarding_intent_targets,
)
from app.domain.profile.models import PreferenceAspect, TasteField, Visibility
from app.domain.profile.onboarding import InterestCategory, OnboardingProfile
from app.domain.recommendation import (
    InvalidSearchIntent,
    RecommendationSignal,
    SearchIntent,
    VectorSpace,
    build_search_queries,
    intent_key,
)
from app.infrastructure.persistence import InMemorySessionStore
from app.infrastructure.persistence.memory_search_intent_store import InMemorySearchIntentStore
from app.infrastructure.persistence.redis_search_intent_store import RedisSearchIntentStore

NOW = datetime(2026, 10, 20, tzinfo=UTC)
OUTDOOR = ("관심사", "아웃도어")
SKI_PAYLOAD = {
    "sense": "겨울 스포츠 스키",
    "productTypes": ["방한 장갑", "넥워머", "고글", "핫팩"],
    "content": "레저·스포츠 스포츠·아웃도어 용품. 스키장에서 쓰는 방한 장갑, 넥워머, 고글.",
    "usage": "스키장에서 스키를 탈 때 추위를 막는 방한 장갑, 넥워머, 핫팩.",
    "gift": "스키를 좋아하는 사람이 받으면 좋은 방한 장갑과 넥워머.",
    "explore": "리빙 침구·욕실. 겨울 야외 활동 뒤 몸을 녹이는 극세사 담요와 온열 소품.",
}
SKI = SearchIntent.from_payload(SKI_PAYLOAD)
# 모델이 돌려주는 모양. 카테고리는 목록에서 고르고, 코드가 content·explore 앞에 붙인다.
MODEL_SKI_PAYLOAD = {
    "sense": SKI_PAYLOAD["sense"],
    "productTypes": SKI_PAYLOAD["productTypes"],
    "category": "레저·스포츠 스포츠·아웃도어 용품",
    "content": "스키장에서 쓰는 방한 장갑, 넥워머, 고글.",
    "usage": SKI_PAYLOAD["usage"],
    "gift": SKI_PAYLOAD["gift"],
    "explore": "겨울 야외 활동 뒤 몸을 녹이는 극세사 담요와 온열 소품.",
    "exploreCategory": "리빙 침구·욕실",
}


def _signal(field: TasteField, value: str, path=OUTDOOR, **extra) -> RecommendationSignal:
    return RecommendationSignal(
        field=field,
        value=value,
        confidence=0.9,
        visibility=Visibility.FRIENDS,
        updated_at=NOW,
        taxonomy_path=path,
        **extra,
    )


# ---- 도메인 ----


def test_intent_key_normalizes_value_and_keeps_the_field() -> None:
    assert intent_key(" 스키 ", OUTDOOR) == intent_key("스키", ["관심사", "아웃도어"])
    # 분야가 다르면 뜻이 다를 수 있어 다른 키다.
    assert intent_key("차", ("관심사", "음식·미식")) != intent_key("차", ("관심사", "자동차"))
    assert intent_key("차", None) == "차|"


@pytest.mark.parametrize(
    "change",
    [
        {"productTypes": ["장갑"]},
        {"productTypes": "장갑, 넥워머"},
        {"content": "   "},
        {"usage": "가" * 161},
        {"sense": None},
        {"explore": ""},
    ],
)
def test_intent_rejects_malformed_payloads(change: dict) -> None:
    with pytest.raises(InvalidSearchIntent):
        SearchIntent.from_payload({**SKI_PAYLOAD, **change})


def test_intent_round_trips_through_payload() -> None:
    assert SearchIntent.from_payload(SKI.to_payload()) == SKI
    assert SKI.text(VectorSpace.CONTENT).startswith("레저·스포츠")
    assert SKI.text(VectorSpace.USAGE).startswith("스키장에서")
    assert SKI.text(VectorSpace.GIFT).startswith("스키를 좋아하는")


def test_schema_requires_every_field_for_strict_structured_output() -> None:
    assert SEARCH_INTENT_SCHEMA["additionalProperties"] is False
    assert set(SEARCH_INTENT_SCHEMA["required"]) == set(SEARCH_INTENT_SCHEMA["properties"])
    categories = SEARCH_INTENT_SCHEMA["properties"]["category"]["enum"]
    assert "리빙 주방·식기" in categories and "레저·스포츠 캠핑 용품" in categories


def test_model_payload_gets_the_catalog_category_as_a_heading() -> None:
    from app.application.search_intent_prompts import intent_payload

    assert SearchIntent.from_payload(intent_payload(MODEL_SKI_PAYLOAD)) == SKI


# ---- 쿼리 ----


def test_queries_use_the_intent_for_its_signal_and_templates_for_the_rest() -> None:
    signals = [
        _signal(TasteField.HOBBIES, "스키"),
        _signal(TasteField.INTERESTS, "커피", path=("관심사", "음식·미식")),
        _signal(
            TasteField.PREFERENCES,
            "미니멀한 디자인",
            path=("취향", "시각", "디자인"),
            aspect=PreferenceAspect.ATTRIBUTE,
        ),
    ]

    queries = build_search_queries(signals, intents={intent_key("스키", OUTDOOR): SKI})
    texts = {(query.signal.value, query.space): query.text for query in queries}

    assert texts[("스키", VectorSpace.USAGE)] == SKI.usage
    assert texts[("스키", VectorSpace.GIFT)] == SKI.gift
    # 의도가 없는 관심사는 템플릿 그대로다.
    assert texts[("커피", VectorSpace.CONTENT)] == "커피 관련 제품"
    # 취향 쿼리는 의도를 쓰지 않는다.
    assert ("미니멀한 디자인", VectorSpace.CONTENT) in texts
    assert "스키 관련 제품 중 미니멀한 디자인" in {query.text for query in queries}


def test_queries_without_intents_keep_the_templates() -> None:
    queries = build_search_queries([_signal(TasteField.HOBBIES, "스키")])
    assert [query.text for query in queries] == [
        "스키 활동에서 사용하는 제품",
        "스키에 관심 있는 사람이 받으면 좋은 선물",
    ]


# ---- 대상 고르기 ----


def test_targets_keep_interest_fields_only_and_dedupe() -> None:
    targets = intent_targets(
        [
            _signal(TasteField.HOBBIES, "스키"),
            _signal(TasteField.INTERESTS, " 스키"),
            _signal(TasteField.PREFERENCES, "혼자 가는 캠핑", aspect=PreferenceAspect.SITUATION),
            _signal(TasteField.DISLIKES, "향 강한 것"),
        ]
    )
    assert targets == [IntentTarget("스키", OUTDOOR)]


def test_onboarding_targets_match_the_recommendation_signal_keys() -> None:
    signals, _ = recommendation_inputs(
        {
            "profile": {field.value: [] for field in TasteField},
            "onboarding": {"interestCategoryCodes": ["COFFEE_TEA", "OUTDOOR"]},
        },
        now=NOW,
    )
    targets = onboarding_intent_targets([InterestCategory.COFFEE_TEA, InterestCategory.OUTDOOR])

    assert [target.key for target in targets] == [target.key for target in intent_targets(signals)]


# ---- 서비스 ----


class IntentGateway:
    def __init__(self, payload=MODEL_SKI_PAYLOAD, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls: list[str] = []

    async def structured(self, messages, *, model, json_schema):
        self.calls.append(json.loads(messages[-1]["content"])["value"])
        assert json_schema is SEARCH_INTENT_SCHEMA
        if self.fail:
            raise RuntimeError("model down")
        return self.payload


def test_prepare_generates_only_missing_intents_and_saves_them() -> None:
    async def scenario() -> None:
        store = InMemorySearchIntentStore()
        await store.save(intent_key("캠핑", OUTDOOR), SKI)
        gateway = IntentGateway()
        service = SearchIntentService(model_gateway=gateway, model="extractor", store=store)

        made = await service.prepare([IntentTarget("캠핑", OUTDOOR), IntentTarget("스키", OUTDOOR)])

        assert made == 1
        assert gateway.calls == ["스키"]
        assert (await service.lookup([IntentTarget("스키", OUTDOOR)])) == {
            intent_key("스키", OUTDOOR): SKI
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "gateway",
    [
        IntentGateway(fail=True),
        IntentGateway({"sense": "x"}),
        # 카테고리가 목록에 없으면 버린다.
        IntentGateway({**MODEL_SKI_PAYLOAD, "category": "레저·스포츠 스키 용품"}),
    ],
)
def test_prepare_swallows_model_failures_and_invalid_output(gateway: IntentGateway) -> None:
    async def scenario() -> None:
        store = InMemorySearchIntentStore()
        service = SearchIntentService(model_gateway=gateway, model="extractor", store=store)

        assert await service.prepare([IntentTarget("스키", OUTDOOR)]) == 0
        assert await store.get_many([intent_key("스키", OUTDOOR)]) == {}

    asyncio.run(scenario())


def test_lookup_returns_nothing_when_the_store_fails() -> None:
    class BrokenStore:
        async def get_many(self, keys):
            raise ConnectionError("redis down")

        async def save(self, key, intent):
            raise AssertionError("not called")

    service = SearchIntentService(model_gateway=IntentGateway(), model="m", store=BrokenStore())
    assert asyncio.run(service.lookup([IntentTarget("스키", OUTDOOR)])) == {}


def test_schedule_runs_in_background_and_skips_keys_in_flight() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()

        class SlowGateway(IntentGateway):
            async def structured(self, messages, *, model, json_schema):
                await gate.wait()
                return await super().structured(messages, model=model, json_schema=json_schema)

        gateway = SlowGateway()
        service = SearchIntentService(
            model_gateway=gateway, model="m", store=InMemorySearchIntentStore()
        )
        service.schedule([IntentTarget("스키", OUTDOOR)])
        await asyncio.sleep(0)
        service.schedule([IntentTarget("스키", OUTDOOR)])
        gate.set()
        await asyncio.gather(*service._tasks)

        assert gateway.calls == ["스키"]
        await service.aclose()

    asyncio.run(scenario())


# ---- Redis 저장소 ----


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiry: dict[str, int] = {}

    async def mget(self, keys: Sequence[str]):
        return [self.values.get(key) for key in keys]

    async def set(self, key: str, value: str, *, ex: int) -> None:
        self.values[key] = value
        self.expiry[key] = ex


def test_redis_store_versions_keys_and_ignores_broken_values() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        store = RedisSearchIntentStore(redis, ttl_seconds=60)  # type: ignore[arg-type]
        key = intent_key("스키", OUTDOOR)
        await store.save(key, SKI)
        redis.values["needu:search-intent:v2:broken|"] = '{"sense": 1}'

        assert list(redis.values)[0] == f"needu:search-intent:v2:{key}"
        assert redis.expiry[f"needu:search-intent:v2:{key}"] == 60
        assert await store.get_many([key, "broken|", "missing|"]) == {key: SKI}

    asyncio.run(scenario())


# ---- 추천과 대화 연결 ----


class RecordingIntents:
    def __init__(self, cached=None) -> None:
        self.cached = cached or {}
        self.scheduled: list[list[IntentTarget]] = []

    async def lookup(self, targets):
        return {
            target.key: self.cached[target.key] for target in targets if target.key in self.cached
        }

    def schedule(self, targets) -> None:
        self.scheduled.append(list(targets))


def test_recommendation_reads_cached_intents_and_schedules_the_missing_ones() -> None:
    seen = {}

    class Engine:
        async def recommend(self, signals, *, guide=None, recipient_name=None, intents=None):
            seen.update(intents)
            return RecommendationBatch(self=(), gift=())

    def item(value: str) -> dict:
        return {
            "value": value,
            "confidence": 0.9,
            "visibility": "friends",
            "updatedAt": NOW.isoformat(),
            "taxonomyPath": list(OUTDOOR),
        }

    profile: dict = {field.value: [] for field in TasteField}
    profile["hobbies"] = [item("스키"), item("등산")]
    intents = RecordingIntents({intent_key("스키", OUTDOOR): SKI})
    service = V1RecommendationService(Engine(), intents=intents)  # type: ignore[arg-type]

    asyncio.run(service.recommend_lists({"profile": profile}))

    assert seen == {intent_key("스키", OUTDOOR): SKI}
    assert intents.scheduled == [[IntentTarget("등산", OUTDOOR)]]


class ChatGateway:
    async def complete(self, messages, *, model):
        return "캠핑 좋죠. 주로 어디로 가요?"

    async def structured(self, messages, *, model, json_schema):
        return {
            "items": [
                {
                    "field": "hobbies",
                    "value": "캠핑",
                    "confidence": 0.9,
                    "evidence": "주말마다 캠핑 가요",
                    "evidenceType": "explicit",
                    "intentType": "both",
                    "deferralReason": None,
                }
            ],
            "axes": [],
            "drop": [],
        }


def test_chat_schedules_intents_for_onboarding_and_new_interests() -> None:
    async def scenario() -> None:
        gateway = ChatGateway()
        intents = RecordingIntents()
        use_cases = ChatUseCases(
            conversations=ConversationService(model_gateway=gateway, extraction_model="e"),
            states=ConversationStateStore(InMemorySessionStore()),
            model_gateway=gateway,
            response_model="r",
            search_intents=intents,  # type: ignore[arg-type]
        )
        await use_cases.create_session(
            user_id=1,
            conversation_room_id=7,
            onboarding=OnboardingProfile(interests=(InterestCategory.OUTDOOR,)),
        )
        await use_cases.send_message(7, "주말마다 캠핑 가요", user_id=1)

        assert [target.value for target in intents.scheduled[0]] == ["캠핑과 아웃도어"]
        assert [target.value for target in intents.scheduled[1]] == ["캠핑"]

    asyncio.run(scenario())
