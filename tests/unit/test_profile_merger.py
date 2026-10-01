from datetime import UTC, datetime, timedelta

from app.domain.profile.merger import ProfileMerger, friend_summary_values
from app.domain.profile.models import (
    DeferralReason,
    DropRef,
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
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
    return ExtractionDelta(items=items)


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


def test_only_top_five_query_signals_are_active_but_candidates_are_preserved() -> None:
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
    assert len(profile.active_signals()) == 5
    assert sum(signal.status == SignalStatus.INACTIVE for signal in profile.signals) == 3


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
    extraction = ExtractionDelta(axes=("손으로 만드는 재미", "모델이 지어낸 기준"))

    result = ProfileMerger().merge(
        ProfileState(),
        extraction,
        utterance="저는 손으로 만드는 재미가 중요한 것 같아요",
        now=NOW,
        source_turn=1,
    )

    assert result.profile.axes == ["손으로 만드는 재미"]


def test_evidence_from_recent_user_turn_creates_new_signal_with_that_turn() -> None:
    result = ProfileMerger().merge(
        ProfileState(),
        delta(item(TasteField.INTERESTS, "퇴근 후 요리", "퇴근하고 요리해요", 0.6)),
        utterance="그 시간이 제일 좋아요",
        now=NOW,
        source_turn=3,
        context_utterances=[(1, "요즘 퇴근하고 요리해요"), (2, "파스타 자주 해요")],
    )

    [signal] = result.accepted
    assert signal.value == "퇴근 후 요리"
    assert signal.source_turn == 1


def test_evidence_only_in_recent_turn_does_not_bump_existing_signal() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.INTERESTS, "요리", "퇴근하고 요리해요")),
        utterance="요즘 퇴근하고 요리해요",
        now=NOW,
        source_turn=1,
    )

    second = merger.merge(
        first.profile,
        delta(item(TasteField.INTERESTS, "요리", "퇴근하고 요리해요")),
        utterance="주말엔 좀 쉬었어요",
        now=NOW,
        source_turn=2,
        context_utterances=[(1, "요즘 퇴근하고 요리해요")],
    )

    assert second.accepted == ()
    [signal] = second.profile.signals
    assert signal.mention_count == 1


def test_evidence_absent_from_all_user_turns_is_rejected() -> None:
    result = ProfileMerger().merge(
        ProfileState(),
        delta(item(TasteField.INTERESTS, "등산", "산에 가요")),
        utterance="그 시간이 제일 좋아요",
        now=NOW,
        source_turn=2,
        context_utterances=[(1, "요즘 퇴근하고 요리해요")],
    )

    assert result.rejected_values == ("등산",)


def test_dislike_supersedes_same_value_liked_before() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.INTERESTS, "러닝", "러닝 해 봤어요")),
        utterance="러닝 해 봤어요",
        now=NOW,
        source_turn=1,
    )
    second = merger.merge(
        first.profile,
        delta(item(TasteField.DISLIKES, "러닝", "러닝은 진짜 싫어요")),
        utterance="러닝은 진짜 싫어요",
        now=NOW,
        source_turn=2,
    )

    active = {(signal.field, signal.value) for signal in second.profile.active_signals()}
    assert active == {(TasteField.DISLIKES, "러닝")}


def test_disliked_value_is_not_counted_as_repeated_interest() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.DISLIKES, "러닝", "러닝은 싫어요")),
        utterance="러닝은 싫어요",
        now=NOW,
        source_turn=1,
    )
    second = merger.merge(
        first.profile,
        delta(
            item(
                TasteField.INTERESTS,
                "러닝",
                "러닝 싫다니까요",
                evidence_type=EvidenceType.INFERRED,
            )
        ),
        utterance="러닝 싫다니까요",
        now=NOW,
        source_turn=2,
    )

    assert second.rejected_values == ("러닝",)
    active = {(signal.field, signal.value) for signal in second.profile.active_signals()}
    assert active == {(TasteField.DISLIKES, "러닝")}


def test_explicit_like_in_current_turn_replaces_old_dislike() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.DISLIKES, "러닝", "러닝은 싫어요")),
        utterance="러닝은 싫어요",
        now=NOW,
        source_turn=1,
    )
    second = merger.merge(
        first.profile,
        delta(item(TasteField.HOBBIES, "러닝", "요즘은 러닝이 좋아졌어요")),
        utterance="요즘은 러닝이 좋아졌어요",
        now=NOW,
        source_turn=5,
    )

    active = {(signal.field, signal.value) for signal in second.profile.active_signals()}
    assert active == {(TasteField.HOBBIES, "러닝")}


def test_same_turn_conflict_keeps_dislike() -> None:
    result = ProfileMerger().merge(
        ProfileState(),
        delta(
            item(TasteField.INTERESTS, "러닝", "러닝은 진짜 싫어요"),
            item(TasteField.DISLIKES, "러닝", "러닝은 진짜 싫어요"),
        ),
        utterance="러닝은 진짜 싫어요",
        now=NOW,
        source_turn=1,
    )

    assert result.rejected_values == ("러닝",)
    active = {(signal.field, signal.value) for signal in result.profile.active_signals()}
    assert active == {(TasteField.DISLIKES, "러닝")}


def test_working_set_spreads_tastes_across_axes() -> None:
    def taste(value: str, path: tuple[str, ...], confidence: float) -> ExtractedItem:
        return ExtractedItem(
            field=TasteField.PREFERENCES,
            value=value,
            confidence=confidence,
            evidence=value,
            evidence_type=EvidenceType.EXPLICIT,
            taxonomy_path=path,
        )

    color = ("취향", "시각", "색상")
    utterance = "무채색 옷, 검은 신발, 회색 가방, 혼자 가는 여행, 조용한 카페"
    result = ProfileMerger().merge(
        ProfileState(),
        delta(
            taste("무채색 옷", color, 0.95),
            taste("검은 신발", color, 0.94),
            taste("회색 가방", color, 0.93),
            taste("혼자 가는 여행", ("취향", "사회", "인원"), 0.8),
            taste("조용한 카페", ("취향", "분위기", "공간"), 0.7),
        ),
        utterance=utterance,
        now=NOW,
        source_turn=1,
    )

    active = [signal.value for signal in result.profile.active_signals()]
    assert active == ["무채색 옷", "혼자 가는 여행", "조용한 카페"]
    stored = next(signal for signal in result.profile.signals if signal.value == "무채색 옷")
    assert stored.taxonomy_path == color
    assert ProfileState.from_dict(result.profile.to_dict()).signals[0].taxonomy_path == color


def test_value_dropped_this_turn_is_not_readded_on_the_same_side() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.HOBBIES, "캠핑", "캠핑 자주 가요")),
        utterance="캠핑 자주 가요",
        now=NOW,
        source_turn=1,
    )
    second = merger.merge(
        first.profile,
        ExtractionDelta(
            items=(
                item(TasteField.INTERESTS, "캠핑", "캠핑은 그냥 해본 말이고"),
                item(TasteField.HOBBIES, "등산", "사실 등산 다녀요"),
            ),
            drop=(DropRef(field=TasteField.HOBBIES, value="캠핑"),),
        ),
        utterance="캠핑은 그냥 해본 말이고 사실 등산 다녀요",
        now=NOW,
        source_turn=2,
    )

    active = {(signal.field, signal.value) for signal in second.profile.active_signals()}
    assert active == {(TasteField.HOBBIES, "등산")}
    assert second.rejected_values == ("캠핑",)


def test_dropping_a_dislike_still_allows_the_opposite_side_in_same_turn() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.DISLIKES, "러닝", "러닝 싫어요")),
        utterance="러닝 싫어요",
        now=NOW,
        source_turn=1,
    )
    second = merger.merge(
        first.profile,
        ExtractionDelta(
            items=(item(TasteField.HOBBIES, "러닝", "사실 러닝 좋아해요"),),
            drop=(DropRef(field=TasteField.DISLIKES, value="러닝"),),
        ),
        utterance="아까 싫다고 한 건 취소요 사실 러닝 좋아해요",
        now=NOW,
        source_turn=2,
    )

    active = {(signal.field, signal.value) for signal in second.profile.active_signals()}
    assert active == {(TasteField.HOBBIES, "러닝")}


def test_superseded_value_is_not_revived_from_previous_turn_context() -> None:
    merger = ProfileMerger()
    first = merger.merge(
        ProfileState(),
        delta(item(TasteField.HOBBIES, "캠핑", "캠핑 자주 가요")),
        utterance="캠핑 자주 가요",
        now=NOW,
        source_turn=1,
    )
    dropped = merger.merge(
        first.profile,
        ExtractionDelta(drop=(DropRef(field=TasteField.HOBBIES, value="캠핑"),)),
        utterance="캠핑은 취소요",
        now=NOW,
        source_turn=2,
    )
    revived = merger.merge(
        dropped.profile,
        delta(item(TasteField.HOBBIES, "캠핑", "캠핑 자주 가요")),
        utterance="요즘 날씨 좋네요",
        now=NOW,
        source_turn=3,
        context_utterances=[(1, "캠핑 자주 가요"), (2, "캠핑은 취소요")],
    )

    assert revived.profile.active_signals() == []
    assert revived.rejected_values == ("캠핑",)
