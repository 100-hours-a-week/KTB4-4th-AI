from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.application.conversation_prompts import (
    FRIEND_SUMMARY_SYSTEM,
    SYSTEM_COMPANION,
    build_friend_summary_messages,
    build_opening_messages,
    build_reply_messages,
)
from app.application.recommendation_service import (
    ONBOARDING_INTEREST_WEIGHT,
    recommendation_inputs,
)
from app.domain.conversation.models import ConversationGoal, ConversationState
from app.domain.profile.models import ProfileState, TasteField
from app.domain.profile.onboarding import (
    Allergy,
    Gender,
    GiftExclusion,
    InterestCategory,
    OnboardingProfile,
)
from app.domain.recommendation import (
    CatalogExclusion,
    CatalogProduct,
    ProductKey,
    build_catalog_exclusion,
    build_search_queries,
)

NOW = datetime(2026, 10, 15, tzinfo=UTC)


def _product(name: str, category: str, subcategory: str | None, document: str | None = None):
    return CatalogProduct(
        key=ProductKey(platform="kakao", external_id=name),
        name=name,
        price=Decimal("10000"),
        category=category,
        subcategory=subcategory,
        content_document=document,
    )


def test_no_onboarding_conditions_exclude_nothing() -> None:
    exclusion = build_catalog_exclusion()

    assert exclusion == CatalogExclusion()
    assert not exclusion.blocks(_product("땅콩 쿠키", "food", "sweets"))


def test_allergy_excludes_risky_and_unclassified_edibles_but_keeps_other_products() -> None:
    exclusion = build_catalog_exclusion(allergies=[Allergy.PEANUT])

    # 땅콩이 흔히 들어가는 세부 카테고리와 무엇인지 모르는 먹거리는 SQL에서 빠진다.
    assert {"food:sweets", "food:bakery", "voucher:dessert", "food:", "voucher:"} <= (
        exclusion.subcategories
    )
    assert exclusion.blocks(_product("수제 쿠키 세트", "food", "sweets"))
    assert exclusion.blocks(_product("이름만 보면 모를 선물세트", "food", None))
    # 세부 카테고리로 안 잡히는 먹거리는 상품명과 content 문서로 거른다.
    assert exclusion.blocks(_product("피넛버터 단백질바", "health", "diet"))
    assert exclusion.blocks(
        _product("아메리카노 + 시즌 케이크", "voucher", "cafe_drink", "땅콩 크림 케이크 세트다.")
    )
    assert not exclusion.blocks(_product("아이스 아메리카노 T", "voucher", "cafe_drink"))
    assert not exclusion.blocks(_product("제주 감귤 3kg", "food", "fruit"))
    # 먹거리가 아니면 성분 단어를 보지 않는다.
    assert not exclusion.blocks(_product("피넛 컬러 니트 파우치", "fashion", "accessories"))


def test_gift_exclusions_map_to_categories_subcategories_and_name_terms() -> None:
    exclusion = build_catalog_exclusion(
        gift_exclusions=[
            GiftExclusion.FOOD,
            GiftExclusion.VOUCHER,
            GiftExclusion.PERFUME,
            GiftExclusion.ALCOHOL,
        ]
    )

    assert exclusion.categories == {"food", "health", "voucher"}
    assert exclusion.blocks(_product("배민상품권 3만원", "voucher", "money_voucher"))
    assert exclusion.blocks(_product("홍삼정 스틱", "health", "traditional_health"))
    assert exclusion.blocks(_product("블루밍 부케 30ml", "beauty", "fragrance"))
    assert exclusion.blocks(_product("오 드 퍼퓸 7ML", "beauty", None))
    assert exclusion.blocks(_product("크리스탈 와인잔 2P", "living", "kitchen"))
    # 향수 제외는 향이 나는 다른 화장품까지 빼지 않는다.
    assert not exclusion.blocks(
        _product("퍼퓸 핸드크림 50ml", "beauty", "body_care", "향수 브랜드의 핸드크림이다.")
    )


def test_cosmetics_and_clothing_exclusions_keep_neighbouring_kinds() -> None:
    exclusion = build_catalog_exclusion(
        gift_exclusions=[GiftExclusion.COSMETICS, GiftExclusion.CLOTHING]
    )

    assert exclusion.blocks(_product("수분 크림", "beauty", "skincare"))
    assert exclusion.blocks(_product("러닝 반팔티", "sports", "sportswear"))
    assert exclusion.blocks(_product("파자마 세트", "fashion", "loungewear"))
    assert not exclusion.blocks(_product("시그니처 향수", "beauty", "fragrance"))
    assert not exclusion.blocks(_product("가죽 카드지갑", "fashion", "wallet"))


def test_recommendation_inputs_turn_onboarding_into_signals_and_exclusion() -> None:
    signals, guide = recommendation_inputs(
        {
            "profile": {field.value: [] for field in TasteField},
            "onboarding": {
                "interestCategoryCodes": ["COFFEE_TEA", "TRAVEL"],
                "allergyCodes": ["PEANUT"],
                "giftExclusionCodes": ["ALCOHOL"],
            },
        },
        now=NOW,
    )

    assert [(signal.field, signal.value, signal.weight) for signal in signals] == [
        (TasteField.INTERESTS, "커피와 차", ONBOARDING_INTEREST_WEIGHT),
        (TasteField.INTERESTS, "여행", ONBOARDING_INTEREST_WEIGHT),
    ]
    assert signals[0].taxonomy_path == ("관심사", "음식·미식")
    assert "food:sweets" in guide.catalog_exclusion.subcategories
    assert "와인" in guide.catalog_exclusion.name_terms
    # 대화 내용이 없어도 고른 관심 카테고리로 쿼리가 만들어진다.
    texts = [query.text for query in build_search_queries(signals)]
    assert "커피와 차 관련 제품" in texts
    assert "여행에 관심 있는 사람이 받으면 좋은 선물" in texts


def test_recommendation_inputs_without_onboarding_only_exclude_pets() -> None:
    signals, guide = recommendation_inputs({"profile": {field.value: [] for field in TasteField}})

    assert signals == ()
    # 반려동물 신호가 없으면 반려동물 대분류만 뺀다. 그 밖의 조건은 없다.
    assert guide.catalog_exclusion == CatalogExclusion(categories=frozenset({"pet"}))


def _profile_with(field: TasteField, value: str, path: list[str] | None) -> dict:
    profile: dict = {item.value: [] for item in TasteField}
    profile[field.value] = [
        {
            "value": value,
            "confidence": 0.8,
            "visibility": "friends",
            "updatedAt": NOW.isoformat(),
            "taxonomyPath": path,
        }
    ]
    return profile


def test_pet_category_stays_only_with_a_pet_signal() -> None:
    pets = ["관심사", "반려동물"]
    walking = recommendation_inputs(
        {"profile": _profile_with(TasteField.HOBBIES, "산책", ["관심사", "운동·피트니스"])}
    )[1]
    lives_with_dog = recommendation_inputs(
        {"profile": _profile_with(TasteField.LIFESTYLE, "반려견과 산다", pets)}
    )[1]
    dislikes_cats = recommendation_inputs(
        {"profile": _profile_with(TasteField.DISLIKES, "고양이 털", pets)}
    )[1]
    chose_pets = recommendation_inputs(
        {
            "profile": {field.value: [] for field in TasteField},
            "onboarding": {"interestCategoryCodes": ["PET"]},
        }
    )[1]

    assert "pet" in walking.catalog_exclusion.categories
    assert "pet" not in lives_with_dog.catalog_exclusion.categories
    # 싫다고 한 것은 반려동물을 원한다는 뜻이 아니다.
    assert "pet" in dislikes_cats.catalog_exclusion.categories
    assert "pet" not in chose_pets.catalog_exclusion.categories


def test_recommendation_inputs_reject_unknown_codes() -> None:
    with pytest.raises(ValueError):
        recommendation_inputs(
            {
                "profile": {field.value: [] for field in TasteField},
                "onboarding": {"allergyCodes": ["SESAME"]},
            }
        )


def test_onboarding_survives_session_state_round_trip() -> None:
    onboarding = OnboardingProfile(
        nickname="니듀",
        birth_date=date(2000, 1, 1),
        gender=Gender.FEMALE,
        interests=(InterestCategory.COFFEE_TEA, InterestCategory.TRAVEL),
        allergies=(Allergy.PEANUT,),
        gift_exclusions=(GiftExclusion.ALCOHOL,),
    )
    state = ConversationState(user_id=1, conversation_room_id=42, onboarding=onboarding)

    restored = ConversationState.from_dict(state.to_dict())

    assert restored.onboarding == onboarding
    # 이전 버전에서 저장한 세션에는 onboarding 키가 없다.
    legacy = state.to_dict()
    del legacy["onboarding"]
    assert ConversationState.from_dict(legacy).onboarding is None


def _named_state(nickname: str | None) -> ConversationState:
    return ConversationState(
        user_id=1,
        conversation_room_id=42,
        onboarding=OnboardingProfile(nickname=nickname),
    )


def test_chat_calls_the_user_by_first_name_only_when_known() -> None:
    named = build_reply_messages(_named_state("서동휘"), ConversationGoal.CHAT, "안녕하세요")
    unnamed = build_reply_messages(_named_state("."), ConversationGoal.CHAT, "안녕하세요")
    opening = build_opening_messages(name="동휘")

    assert named[0]["content"] == SYSTEM_COMPANION
    assert '"동휘님"이라고 부른다' in named[1]["content"]
    assert not any('님"이라고 부른다' in message["content"] for message in unnamed)
    assert '"동휘님"이라고 부른다' in opening[-1]["content"]


def test_summary_uses_the_full_username() -> None:
    named = build_friend_summary_messages(ProfileState(), name="서동휘")[0]["content"]
    unnamed = build_friend_summary_messages(ProfileState())[0]["content"]

    assert '첫 문장은 "니쥬가 보기에 서동휘님은"으로 시작해서' in named
    assert '"사용자", "이분", "고객" 같은 다른 호칭은 쓰지 않는다' in named
    assert "니쥬가 보기에 사용자는" not in named
    # 이름이 없으면 지금까지와 같은 프롬프트다.
    assert unnamed == FRIEND_SUMMARY_SYSTEM


@pytest.mark.parametrize(
    "written",
    [
        "니쥬가 보기에 서동휘님은 캠핑을 즐겨요.",
        "니쥬가 보기에 동휘님은 캠핑을 즐겨요.",
        "니쥬가 보기에 사용자는 캠핑을 즐겨요.",
        "니쥬가 보기에 서동휘 님이 캠핑을 즐겨요.",
        "니쥬가 보기에 서동휘님님은 캠핑을 즐겨요.",
        "니쥬가 보기에 동휘님, 캠핑을 즐겨요.",
        "서동휘님은 캠핑을 즐겨요.",
        "동휘님은 캠핑을 즐겨요.",
    ],
)
def test_summary_intro_is_fixed_to_full_username(written: str) -> None:
    from app.domain.conversation.guards import fix_summary_intro

    assert fix_summary_intro(written, name="서동휘", address="동휘") == (
        "니쥬가 보기에 서동휘님은 캠핑을 즐겨요."
    )


def test_summary_intro_guard_leaves_unrecognizable_sentences_alone() -> None:
    from app.domain.conversation.guards import fix_summary_intro

    # 주어를 알아볼 수 없으면 고치다 문장이 깨질 수 있어 그대로 둔다.
    for text in ("니쥬가 보기에 커피를 좋아하는 사람이에요.", "캠핑을 즐기는 분입니다."):
        assert fix_summary_intro(text, name="서동휘", address="동휘") == text
    # 이름이 없으면 "사용자는"으로 맞춘다.
    assert fix_summary_intro("니쥬가 보기에 이분은 캠핑을 즐겨요.") == (
        "니쥬가 보기에 사용자는 캠핑을 즐겨요."
    )
