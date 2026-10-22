import json
from datetime import UTC, datetime

from app.application.conversation_prompts import build_friend_summary_messages
from app.domain.profile.keywords import composed_keywords, fallback_keywords
from app.domain.profile.models import (
    EvidenceType,
    LinkRole,
    ProfileSignal,
    ProfileState,
    TasteField,
    Visibility,
)

NOW = datetime(2026, 10, 22, tzinfo=UTC)
SCENT = ("취향", "후각", "향")


def signal(
    field: TasteField,
    value: str,
    *,
    target: str | None = None,
    path: tuple[str, ...] | None = None,
    confidence: float = 0.8,
) -> ProfileSignal:
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=confidence,
        link_role=LinkRole.WEIGHT if field == TasteField.PREFERENCES else LinkRole.QUERY,
        visibility=Visibility.FRIENDS,
        intent_type=None,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=NOW,
        updated_at=NOW,
        target=target,
        taxonomy_path=path,
    )


TASTES = [
    signal(TasteField.PREFERENCES, "향이 좋은 바디로션", target="바디로션", path=SCENT),
    signal(TasteField.PREFERENCES, "포근한 향 바디로션", target="바디로션", path=SCENT),
    signal(TasteField.PREFERENCES, "메뉴만 보고 고르기", path=("취향", "선택", "기준")),
]


def test_composed_taste_keyword_keeps_only_the_quality() -> None:
    keywords = composed_keywords(
        [
            {"label": "포근한 향", "sources": [0, 1]},
            {"label": "포근한 향 바디로션", "sources": [1]},
        ],
        TASTES,
        taste=True,
    )

    assert [keyword.label for keyword in keywords] == ["포근한 향"]
    assert keywords[0].score == 0.8


def test_composed_taste_strips_a_leftover_target_instead_of_dropping() -> None:
    keywords = composed_keywords(
        [{"label": "가사 중심 노래", "sources": [0]}],
        [signal(TasteField.PREFERENCES, "가사 먼저 보고 고르는 노래", target="노래")],
        taste=True,
    )

    assert [keyword.label for keyword in keywords] == ["가사 중심"]


def test_composed_taste_drops_label_left_with_only_a_modifier() -> None:
    keywords = composed_keywords(
        [{"label": "맛있는 떡볶이", "sources": [0]}],
        [signal(TasteField.PREFERENCES, "맛있는 엽떡 시그니처 떡볶이", target="떡볶이")],
        taste=True,
    )

    assert keywords == []


def test_composed_taste_keeps_one_keyword_per_axis() -> None:
    keywords = composed_keywords(
        [
            {"label": "포근한 향", "sources": [1]},
            {"label": "좋은 향", "sources": [0]},
            {"label": "메뉴 보고 고르기", "sources": [2]},
        ],
        TASTES,
        taste=True,
    )

    assert [keyword.label for keyword in keywords] == ["포근한 향", "메뉴 보고 고르기"]


def test_composed_keywords_drop_items_without_valid_sources() -> None:
    interests = [
        signal(TasteField.INTERESTS, "디저트"),
        signal(TasteField.INTERESTS, "케이크 맛집"),
    ]

    keywords = composed_keywords(
        [
            {"label": "디저트 맛집", "sources": [0, 1]},
            {"label": "빵", "sources": [7]},
            {"label": "카페", "sources": []},
            {"label": "아주아주아주긴관심사이름입니다", "sources": [0]},
            "바디케어",
        ],
        interests,
        taste=False,
    )

    assert [keyword.label for keyword in keywords] == ["디저트 맛집"]


def test_fallback_taste_strips_target_and_dedupes_axis() -> None:
    keywords = fallback_keywords([TASTES[1], TASTES[0], TASTES[2]], taste=True)

    assert [keyword.label for keyword in keywords] == ["포근한 향", "메뉴만 보고 고르기"]


def test_summary_input_carries_dislikes_only_as_avoids() -> None:
    profile = ProfileState(
        signals=[
            signal(TasteField.INTERESTS, "캠핑"),
            signal(TasteField.DISLIKES, "강한 향", confidence=0.9),
        ]
    )

    content = json.loads(build_friend_summary_messages(profile)[1]["content"])

    assert content["avoids"] == [{"value": "강한 향"}]
    values = [item["value"] for group in content["interests"] for item in group["items"]]
    assert values == ["캠핑"]
