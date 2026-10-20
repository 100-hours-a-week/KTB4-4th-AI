import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from app.application.recommendation_engine import EXPLORE_TOP_K, RecommendationEngine
from app.domain.profile.models import TasteField, Visibility
from app.domain.recommendation import (
    CatalogProduct,
    CatalogSearchResult,
    ProductKey,
    RecommendationMode,
    RecommendationSignal,
    SearchIntent,
    SearchQuery,
    VectorMatch,
    VectorSpace,
    build_explore_queries,
    build_search_queries,
    intent_key,
    rank_recommendations,
)
from app.domain.recommendation.ranking import MAX_EXPLORE
from app.domain.recommendation.reasons import recommendation_reason

NOW = datetime(2026, 10, 20, tzinfo=UTC)
FOOD = ("관심사", "음식·미식")
SPORTS = ("관심사", "운동·피트니스")


def _signal(field, value, path, *, visibility=Visibility.FRIENDS) -> RecommendationSignal:
    return RecommendationSignal(
        field=field,
        value=value,
        confidence=0.9,
        visibility=visibility,
        updated_at=NOW,
        taxonomy_path=path,
    )


def _intent(name: str, explore: str) -> SearchIntent:
    return SearchIntent(
        sense=name,
        product_types=(f"{name} 용품", f"{name} 소품"),
        content=f"{name} content",
        usage=f"{name} usage",
        gift=f"{name} gift",
        explore=explore,
    )


def _product(external_id: str, name: str, category: str = "living") -> CatalogProduct:
    return CatalogProduct(
        key=ProductKey(platform="kakao", external_id=external_id),
        name=name,
        price=Decimal("10000.00"),
        category=category,
        gift_available=True,
    )


COFFEE = _signal(TasteField.INTERESTS, "커피", FOOD)
BOWLING = _signal(TasteField.HOBBIES, "볼링", SPORTS)
INTENTS = {
    intent_key("커피", FOOD): _intent("커피", "커피 explore"),
    intent_key("볼링", SPORTS): _intent("볼링", "볼링 explore"),
}


# ---- 탐색 쿼리 ----


def test_explore_queries_are_one_per_signal_with_an_intent() -> None:
    camping = _signal(TasteField.HOBBIES, "캠핑", ("관심사", "아웃도어"))
    queries = build_search_queries([COFFEE, BOWLING, camping], intents=INTENTS)

    explore = build_explore_queries(queries, intents=INTENTS)

    # 사용자 취향은 붙이지 않는다. 의도가 없는 캠핑은 탐색 쿼리가 없다.
    assert [(query.signal.value, query.text) for query in explore] == [
        ("커피", "커피 explore"),
        ("볼링", "볼링 explore"),
    ]
    assert all(query.exploration and query.space == VectorSpace.CONTENT for query in explore)
    assert build_explore_queries(queries, intents=None) == ()


# ---- 추천 이유 ----


def _scored(
    query: SearchQuery,
    product: CatalogProduct,
    contribution: float = 0.5,
    similarity: float = 0.7,
):
    from app.domain.recommendation.models import ScoredMatch

    return ScoredMatch(
        match=VectorMatch(query=query, product=product, similarity=similarity),
        contribution=contribution,
    )


def _query(signal, *, exploration=False, query_id="q_1") -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        signal=signal,
        modes=frozenset({RecommendationMode.SELF, RecommendationMode.GIFT}),
        space=VectorSpace.CONTENT,
        text="t",
        exploration=exploration,
    )


def test_explore_reason_says_it_is_a_new_suggestion() -> None:
    product = _product("1", "보드게임")
    explore = [_scored(_query(BOWLING, exploration=True), product)]
    private = [
        _scored(
            _query(
                _signal(TasteField.HOBBIES, "볼링", SPORTS, visibility=Visibility.PRIVATE),
                exploration=True,
            ),
            product,
        )
    ]

    assert (
        recommendation_reason(explore, RecommendationMode.SELF)
        == "볼링을 좋아하는 취향을 반영한 새로운 추천이에요."
    )
    assert (
        recommendation_reason(explore, RecommendationMode.GIFT, recipient_name="서동휘")
        == "볼링을 좋아하는 서동휘님에게 새롭게 제안하는 상품이에요."
    )
    # 친구에게 보이지 않는 관심사는 이름을 쓰지 않는다.
    assert (
        recommendation_reason(private, RecommendationMode.GIFT, recipient_name="서동휘")
        == "서동휘님의 취향과 관심사를 반영한 새로운 추천이에요."
    )


def test_reason_names_two_interests_and_the_boosting_taste() -> None:
    product = _product("1", "캠핑용 드립 세트")
    camping = _signal(TasteField.HOBBIES, "캠핑", ("관심사", "아웃도어"))
    matches = [
        _scored(_query(COFFEE), product, 0.6),
        _scored(_query(camping, query_id="q_2"), product, 0.5),
    ]

    assert (
        recommendation_reason(matches, RecommendationMode.SELF, boost_label="미니멀한 디자인")
        == "커피와 캠핑을 함께 즐기는 취향에 맞는 상품이에요. 미니멀한 디자인 취향도 반영했어요."
    )
    assert (
        recommendation_reason(matches, RecommendationMode.GIFT, recipient_name="서동휘")
        == "커피와 캠핑을 함께 좋아하는 서동휘님에게 잘 어울리는 선물이에요."
    )


def test_weak_second_match_does_not_make_a_shared_reason() -> None:
    product = _product("1", "디지털카메라")
    camping = _signal(TasteField.HOBBIES, "캠핑", ("관심사", "아웃도어"))
    matches = [
        _scored(_query(COFFEE), product, 0.6, similarity=0.62),
        _scored(_query(camping, query_id="q_2"), product, 0.5, similarity=0.55),
    ]

    assert (
        recommendation_reason(matches, RecommendationMode.SELF)
        == "커피에 대한 관심과 잘 맞는 상품이에요."
    )


def test_reason_varies_by_rank_and_picks_particles_by_final_consonant() -> None:
    product = _product("1", "랜턴")
    camping = [_scored(_query(_signal(TasteField.HOBBIES, "캠핑", None)), product)]
    running = [_scored(_query(_signal(TasteField.HOBBIES, "러닝", None)), product)]
    tea = [_scored(_query(_signal(TasteField.WANTS, "티포트", None)), product)]

    first = recommendation_reason(camping, RecommendationMode.SELF, rank=0)
    second = recommendation_reason(camping, RecommendationMode.SELF, rank=1)
    assert first == "캠핑에 대한 관심과 잘 맞는 상품이에요."
    assert second == "캠핑을 즐길 때 챙기기 좋은 상품이에요."
    assert recommendation_reason(running, RecommendationMode.SELF, rank=1).startswith("러닝을")
    assert (
        recommendation_reason(tea, RecommendationMode.SELF)
        == "갖고 싶다고 한 티포트와 가까운 상품이에요."
    )


# ---- 자리 배분 ----


def _matches(query: SearchQuery, prefix: str, top: float, count: int):
    return [
        VectorMatch(
            query=query, product=_product(f"{prefix}-{index}", prefix), similarity=top - index / 100
        )
        for index in range(count)
    ]


def test_explore_items_take_reserved_slots_after_matched_items() -> None:
    coffee = _query(COFFEE)
    bowling_explore = _query(BOWLING, exploration=True, query_id="e_1")
    knitting_explore = _query(
        _signal(TasteField.HOBBIES, "뜨개질", None), exploration=True, query_id="e_2"
    )

    ranked = rank_recommendations(
        [
            *_matches(coffee, "coffee", 0.6, 25),
            # 탐색 상품의 유사도가 더 높아도 관심사 상품 뒤에 온다.
            *_matches(bowling_explore, "board", 0.9, 5),
            *_matches(knitting_explore, "diy", 0.9, 5),
        ],
        mode=RecommendationMode.SELF,
        now=NOW,
    )

    owners = [item.product.key.external_id.split("-")[0] for item in ranked]
    assert len(ranked) == 20
    assert owners[:16] == ["coffee"] * 16
    # 관심사 하나당 2개, 목록 전체 4개까지다.
    assert sorted(owners[16:]) == ["board", "board", "diy", "diy"]
    assert len(owners[16:]) == MAX_EXPLORE
    scores = [item.score for item in ranked]
    assert scores == sorted(scores, reverse=True)
    assert "새로운 추천" in ranked[-1].reason


def test_explore_fills_a_short_list_up_to_ten() -> None:
    coffee = _query(COFFEE)
    bowling_explore = _query(BOWLING, exploration=True, query_id="e_1")

    ranked = rank_recommendations(
        [*_matches(coffee, "coffee", 0.6, 3), *_matches(bowling_explore, "board", 0.7, 10)],
        mode=RecommendationMode.SELF,
        now=NOW,
    )

    owners = [item.product.key.external_id.split("-")[0] for item in ranked]
    assert owners == ["coffee"] * 3 + ["board"] * 7


def test_a_product_also_found_by_an_interest_query_is_a_matched_item() -> None:
    shared = _product("shared", "드립 세트")
    ranked = rank_recommendations(
        [
            VectorMatch(query=_query(COFFEE), product=shared, similarity=0.6),
            VectorMatch(
                query=_query(BOWLING, exploration=True, query_id="e_1"),
                product=shared,
                similarity=0.9,
            ),
        ],
        mode=RecommendationMode.SELF,
        now=NOW,
    )

    assert len(ranked) == 1
    assert "새로운 추천" not in ranked[0].reason


# ---- 엔진 2단계 ----


class TextEmbedder:
    space_id = "test"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed_passages(self, texts):
        raise AssertionError("not used")

    async def embed_queries(self, texts: Sequence[str]):
        start = len(self.texts)
        self.texts.extend(texts)
        return [[float(start + index)] for index in range(len(texts))]


class TextCatalog:
    """검색한 쿼리 문장에 따라 결과를 돌려준다. 커피는 상품이 있고 볼링은 없는 카탈로그다."""

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder
        self.searched: list[tuple[str, VectorSpace, int]] = []

    async def ping(self) -> bool:
        return True

    async def search(self, *, vector, space, embedding_space_id, limit, **_):
        text = self.embedder.texts[int(vector[0])]
        self.searched.append((text, space, limit))
        if text.startswith("커피"):
            return [
                CatalogSearchResult(product=_product(f"coffee-{i}", "원두"), similarity=0.75)
                for i in range(3)
            ]
        if text == "볼링 usage":
            return [CatalogSearchResult(product=_product("glove", "골프 장갑"), similarity=0.58)]
        if "미니멀한 디자인" in text and "explore" not in text:
            return [CatalogSearchResult(product=_product("rug", "미니러그"), similarity=0.60)]
        if text.startswith("볼링 explore"):
            return [CatalogSearchResult(product=_product("board", "보드게임"), similarity=0.70)]
        return []


def test_engine_replaces_an_uncovered_interest_with_exploration() -> None:
    embedder = TextEmbedder()
    catalog = TextCatalog(embedder)
    engine = RecommendationEngine(embedder=embedder, catalog=catalog)

    batch = asyncio.run(engine.recommend([COFFEE, BOWLING], now=NOW, intents=INTENTS))

    names = [item.product.name for item in batch.self]
    assert "골프 장갑" not in names
    assert names[-1] == "보드게임"
    assert batch.self[-1].reason == "볼링을 좋아하는 취향을 반영한 새로운 추천이에요."
    # 탐색 검색은 상품이 없는 볼링만 content 공간에서 한다.
    explore_searches = [entry for entry in catalog.searched if "explore" in entry[0]]
    assert explore_searches == [("볼링 explore", VectorSpace.CONTENT, EXPLORE_TOP_K)]


def test_engine_without_intents_does_not_explore() -> None:
    embedder = TextEmbedder()
    catalog = TextCatalog(embedder)
    engine = RecommendationEngine(embedder=embedder, catalog=catalog)

    asyncio.run(engine.recommend([COFFEE, BOWLING], now=NOW))

    assert not any("explore" in text for text, _, _ in catalog.searched)


def test_taste_queries_anchored_on_an_uncovered_interest_are_dropped_too() -> None:
    from app.domain.profile.models import PreferenceAspect

    minimal = RecommendationSignal(
        field=TasteField.PREFERENCES,
        value="미니멀한 디자인",
        confidence=0.8,
        visibility=Visibility.FRIENDS,
        updated_at=NOW,
        aspect=PreferenceAspect.ATTRIBUTE,
        taxonomy_path=("취향", "시각", "디자인"),
    )
    embedder = TextEmbedder()
    catalog = TextCatalog(embedder)
    engine = RecommendationEngine(embedder=embedder, catalog=catalog)

    batch = asyncio.run(engine.recommend([BOWLING, minimal], now=NOW, intents=INTENTS))

    # "볼링 관련 제품 중 미니멀한 디자인"은 검색됐지만 볼링이 상품 없는 관심사라 결과를 버린다.
    assert any(text == "볼링 관련 제품 중 미니멀한 디자인" for text, _, _ in catalog.searched)
    assert [item.product.name for item in batch.self] == ["보드게임"]


def test_taste_reason_does_not_repeat_the_word_taste() -> None:
    product = _product("1", "땅콩 버터")
    peanut = _signal(TasteField.PREFERENCES, "땅콩만 좋아하는 취향", None)
    minimal = _signal(TasteField.PREFERENCES, "미니멀한 디자인", None)
    matches = [_scored(_query(peanut), product)]

    assert (
        recommendation_reason(matches, RecommendationMode.GIFT, recipient_name="서동휘")
        == "서동휘님의 땅콩만 좋아하는 취향에 맞춰 선물하기 좋은 상품이에요."
    )
    assert (
        recommendation_reason(matches, RecommendationMode.SELF)
        == "땅콩만 좋아하는 취향과 잘 맞는 상품이에요."
    )
    assert (
        recommendation_reason([_scored(_query(minimal), product)], RecommendationMode.SELF)
        == "미니멀한 디자인 취향과 잘 맞는 상품이에요."
    )
    coffee = [_scored(_query(COFFEE), product)]
    assert recommendation_reason(
        coffee, RecommendationMode.SELF, boost_label="고소한 맛 취향"
    ).endswith("고소한 맛 취향도 반영했어요.")
