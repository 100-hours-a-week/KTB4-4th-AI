from datetime import UTC, datetime, timedelta

from app.domain.profile.merger import ProfileMerger, friend_summary_values
from app.domain.profile.models import (
    AssessmentStatus,
    DeferralReason,
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
    GoalAssessment,
    ProfileState,
    SignalStatus,
    TasteField,
)

NOW = datetime(2026, 9, 18, tzinfo=UTC)


def item(
    field: TasteField,
    value: str,
    evidence: str,
    confidence: float = 0.9,
    *,
    evidence_type: EvidenceType = EvidenceType.EXPLICIT,
    deferral_reason: DeferralReason | None = None,
) -> ExtractedItem:
    return ExtractedItem(
        field=field,
        value=value,
        confidence=confidence,
        evidence=evidence,
        evidence_type=evidence_type,
        deferral_reason=deferral_reason,
    )


def delta(*items: ExtractedItem) -> ExtractionDelta:
    return ExtractionDelta(
        items=items,
        goal_assessment=GoalAssessment("INTEREST", AssessmentStatus.FOUND),
    )


def test_rejects_signal_without_user_evidence_and_generic_value() -> None:
    result = ProfileMerger().merge(
        ProfileState(),
        delta(
            item(TasteField.INTERESTS, "캠핑", "사용자가 하지 않은 말"),
            item(TasteField.INTERESTS, "아무거나", "아무거나"),
        ),
        utterance="아무거나 괜찮아요",
        now=NOW,
        source_turn=1,
    )

    assert result.profile.active_signals() == []
    assert result.rejected_values == ("캠핑", "아무거나")


def test_merges_duplicate_signal_and_preserves_first_seen_time() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.HOBBIES, "핸드 드립", "핸드 드립 좋아해요", 0.7)),
        utterance="핸드 드립 좋아해요",
        now=NOW,
        source_turn=1,
    ).profile
    second = merger.merge(
        first,
        delta(item(TasteField.HOBBIES, "핸드 드립", "핸드 드립 자주 해요", 0.9)),
        utterance="핸드 드립 자주 해요",
        now=NOW + timedelta(minutes=1),
        source_turn=2,
    ).profile

    active = second.active_signals()
    assert len(active) == 1
    assert active[0].confidence == 0.9
    assert active[0].mention_count == 2
    assert active[0].first_seen_at == NOW
    assert active[0].updated_at == NOW + timedelta(minutes=1)


def test_owned_supersedes_matching_want() -> None:
    merger = ProfileMerger()
    wanted = merger.merge(
        ProfileState(),
        delta(item(TasteField.WANTS, "티타늄 컵", "티타늄 컵 갖고 싶어요")),
        utterance="티타늄 컵 갖고 싶어요",
        now=NOW,
        source_turn=1,
    ).profile
    owned = merger.merge(
        wanted,
        delta(item(TasteField.OWNED, "티타늄 컵", "티타늄 컵 결국 샀어요")),
        utterance="티타늄 컵 결국 샀어요",
        now=NOW + timedelta(minutes=1),
        source_turn=2,
    ).profile

    statuses = {signal.field: signal.status for signal in owned.signals}
    assert statuses[TasteField.WANTS] == SignalStatus.SUPERSEDED
    assert statuses[TasteField.OWNED] == SignalStatus.ACTIVE


def test_only_top_six_query_signals_are_active_but_candidates_are_preserved() -> None:
    merger = ProfileMerger()
    profile = ProfileState()
    for index in range(8):
        value = f"취미{index}"
        profile = merger.merge(
            profile,
            delta(item(TasteField.HOBBIES, value, value, 0.5 + index * 0.05)),
            utterance=value,
            now=NOW + timedelta(minutes=index),
            source_turn=index + 1,
        ).profile

    assert len(profile.signals) == 8
    assert len(profile.active_signals()) == 6
    assert sum(signal.status == SignalStatus.INACTIVE for signal in profile.signals) == 2


def test_friend_summary_projection_excludes_sensitive_fields() -> None:
    merger = ProfileMerger()
    profile = merger.merge(
        ProfileState(),
        delta(
            item(TasteField.INTERESTS, "캠핑", "캠핑 좋아해요"),
            item(
                TasteField.UNAFFORDABLE,
                "비싼 텐트",
                "비싼 텐트는 못 샀어요",
                deferral_reason=DeferralReason.PRICE,
            ),
            item(TasteField.CONSTRAINTS, "견과류 알레르기", "견과류 알레르기 있어요"),
        ),
        utterance="캠핑 좋아해요. 비싼 텐트는 못 샀어요. 견과류 알레르기 있어요",
        now=NOW,
        source_turn=1,
    ).profile

    values = friend_summary_values(profile)

    assert values["interests"] == ["캠핑"]
    assert "비싼 텐트" not in str(values)
    assert "견과류 알레르기" not in str(values)
    disliked_or_constrained = [
        signal
        for signal in profile.active_signals()
        if signal.field in {TasteField.DISLIKES, TasteField.CONSTRAINTS}
    ]
    assert all(signal.visibility.value == "private" for signal in disliked_or_constrained)


def test_axis_must_be_grounded_in_current_user_utterance() -> None:
    extraction = ExtractionDelta(
        axes=("손으로 만드는 재미", "모델이 지어낸 기준"),
        goal_assessment=GoalAssessment("DEEPEN", AssessmentStatus.FOUND),
    )

    result = ProfileMerger().merge(
        ProfileState(),
        extraction,
        utterance="저는 손으로 만드는 재미가 중요한 것 같아요",
        now=NOW,
        source_turn=1,
    )

    assert result.profile.axes == ["손으로 만드는 재미"]
