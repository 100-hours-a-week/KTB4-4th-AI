from datetime import UTC, datetime

from app.domain.conversation.models import ConversationState, SessionStatus
from app.domain.conversation.policy import conversation_progress, recommendation_readiness
from app.domain.profile.models import (
    EvidenceType,
    IntentType,
    LinkRole,
    ProfileSignal,
    SignalStatus,
    TasteField,
    Visibility,
)


def signal(field: TasteField, value: str, confidence: float = 0.9) -> ProfileSignal:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=confidence,
        link_role=(
            LinkRole.QUERY
            if field
            in {
                TasteField.INTERESTS,
                TasteField.HOBBIES,
                TasteField.WANTS,
                TasteField.UNAFFORDABLE,
                TasteField.CONSUMABLES,
            }
            else LinkRole.WEIGHT
        ),
        visibility=Visibility.FRIENDS,
        intent_type=IntentType.BOTH,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=now,
        updated_at=now,
        status=SignalStatus.ACTIVE,
    )


def state() -> ConversationState:
    return ConversationState(user_id=1, conversation_room_id=101)


def taste(value: str, path: tuple[str, ...]) -> ProfileSignal:
    preference = signal(TasteField.PREFERENCES, value)
    preference.taxonomy_path = path
    return preference


def two_taste_axes() -> list[ProfileSignal]:
    return [
        taste("혼자 가는 캠핑", ("취향", "사회", "인원")),
        taste("무채색 장비", ("취향", "시각", "색상")),
    ]


def six_interests() -> list[ProfileSignal]:
    return [
        signal(TasteField.INTERESTS, "캠핑"),
        signal(TasteField.HOBBIES, "핸드드립"),
        signal(TasteField.INTERESTS, "재즈"),
        signal(TasteField.WANTS, "가벼운 컵", confidence=0.5),
        signal(TasteField.INTERESTS, "사진", confidence=0.5),
        signal(TasteField.HOBBIES, "러닝", confidence=0.5),
    ]


def test_ready_profile_does_not_require_dislikes() -> None:
    conversation = state()
    conversation.profile.signals.extend([*six_interests(), *two_taste_axes()])

    readiness = recommendation_readiness(conversation)

    assert readiness.sufficient is True
    assert readiness.missing_signals == ()


def test_one_interest_lists_every_missing_signal() -> None:
    conversation = state()
    conversation.profile.signals.append(signal(TasteField.INTERESTS, "캠핑", confidence=0.5))

    assert recommendation_readiness(conversation).missing_signals == (
        "query_signals",
        "high_confidence_query_signals",
        "taste_axes",
        "gear",
    )


def test_readiness_is_recomputed_when_a_taste_is_replaced() -> None:
    conversation = state()
    tastes = two_taste_axes()
    conversation.profile.signals.extend([*six_interests(), *tastes])
    assert recommendation_readiness(conversation).sufficient is True

    tastes[1].status = SignalStatus.SUPERSEDED
    readiness = recommendation_readiness(conversation)

    assert readiness.sufficient is False
    assert readiness.missing_signals == ("taste_axes",)


def test_repeated_interest_counts_as_high_confidence() -> None:
    conversation = state()
    interests = six_interests()
    conversation.profile.signals.extend([*interests, *two_taste_axes()])
    interests[2].confidence = 0.5
    assert recommendation_readiness(conversation).missing_signals == (
        "high_confidence_query_signals",
    )

    # 여러 번 말한 관심사는 신뢰도가 낮게 매겨졌어도 확실한 것으로 센다.
    interests[3].mention_count = 3
    assert recommendation_readiness(conversation).sufficient is True


def test_owned_item_alone_counts_as_gear() -> None:
    conversation = state()
    conversation.profile.signals.append(signal(TasteField.OWNED, "캠핑 의자"))

    assert "gear" not in recommendation_readiness(conversation).missing_signals


def test_progress_scales_filled_signals_to_100_without_dislikes() -> None:
    conversation = state()
    conversation.profile.signals.extend(
        [
            signal(TasteField.INTERESTS, "캠핑", confidence=0.9),
            signal(TasteField.HOBBIES, "핸드드립", confidence=0.5),
        ]
    )
    # (관심사 2/6 * 25 + 확신 높은 관심사 1/3 * 25) / 85 * 100
    assert conversation_progress(conversation) == 20

    conversation.profile.signals.append(signal(TasteField.OWNED, "캠핑 의자"))
    # 소지품 15 추가
    assert conversation_progress(conversation) == 37

    conversation.profile.signals.extend([*six_interests(), *two_taste_axes()])
    assert conversation_progress(conversation) == 100


def test_progress_does_not_go_down() -> None:
    conversation = state()
    conversation.progress_floor = 60

    assert conversation_progress(conversation) == 60


def test_progress_is_complete_when_input_is_locked_or_under_review() -> None:
    conversation = state()
    conversation.status = SessionStatus.INPUT_LOCKED
    assert conversation_progress(conversation) == 100

    conversation.status = SessionStatus.REVIEW
    assert conversation_progress(conversation) == 100
