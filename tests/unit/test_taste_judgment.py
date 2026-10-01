from datetime import UTC, datetime

from app.application.taste_judgment import (
    Candidate,
    CandidateKind,
    build_judgment_request,
    correction_targets,
    judgment_delta,
    parse_candidates,
    remention_candidates,
    split_sentences,
)
from app.domain.conversation.models import ConversationState, ConversationTurn
from app.domain.profile.merger import ProfileMerger
from app.domain.profile.models import (
    DeferralReason,
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
    PreferenceAspect,
    ProfileState,
    TasteField,
)

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def choice(value: str, confidence: float = 0.95) -> dict[str, object]:
    return {
        "type": "choice",
        "choice": value,
        "confidence": confidence,
        "probabilities": {value: confidence},
    }


def noul(value: float) -> dict[str, object]:
    return {"type": "noul", "noul": value}


def test_parse_candidates_drops_invalid_and_duplicate_and_strips_attribute_fields() -> None:
    candidates, axes, dropped = parse_candidates(
        {
            "candidates": [
                {"subject": "러닝", "kind": "activity", "aspect": "sensory", "target": "운동"},
                {"subject": "러닝", "kind": "activity"},
                {"subject": "고소한 커피", "kind": "attribute", "aspect": "sensory", "target": " "},
                {"subject": "", "kind": "thing"},
                {"subject": "우주선", "kind": "vehicle"},
            ],
            "axes": ["가벼운 게 제일 중요해요"],
        }
    )

    assert candidates == [
        Candidate("러닝", CandidateKind.ACTIVITY),
        Candidate("고소한 커피", CandidateKind.ATTRIBUTE, PreferenceAspect.SENSORY, None),
    ]
    assert axes == ["가벼운 게 제일 중요해요"]
    assert len(dropped) == 2


def test_remention_adds_profile_values_found_in_utterance() -> None:
    profile = (
        ProfileMerger()
        .merge(
            ProfileState(),
            ExtractionDelta(
                items=(
                    ExtractedItem(
                        field=TasteField.HOBBIES,
                        value="러닝",
                        confidence=0.9,
                        evidence="러닝 매일 해요",
                        evidence_type=EvidenceType.EXPLICIT,
                    ),
                )
            ),
            utterance="러닝 매일 해요",
            now=NOW,
            source_turn=1,
        )
        .profile
    )

    found = remention_candidates(profile, "러닝은 이제 좀 질렸어요", [])

    assert found == [Candidate("러닝", CandidateKind.ACTIVITY)]
    assert remention_candidates(profile, "러닝은 이제 좀 질렸어요", found) == []


def test_split_sentences_keeps_korean_sentences() -> None:
    assert split_sentences("러닝은 진짜 싫어요. 숨차서요! 그냥 걷는 건 괜찮아요") == [
        "러닝은 진짜 싫어요.",
        "숨차서요!",
        "그냥 걷는 건 괜찮아요",
    ]


def test_judgment_request_asks_questions_that_fit_each_kind() -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    state.history.extend(
        [
            ConversationTurn(
                role="assistant", content="요즘 즐겨 하는 운동 있어요?", created_at=NOW
            ),
        ]
    )
    request = build_judgment_request(
        state,
        "러닝은 진짜 싫어요. 에어팟 맥스는 갖고 싶어요",
        [
            Candidate("러닝", CandidateKind.ACTIVITY),
            Candidate("에어팟 맥스", CandidateKind.THING),
        ],
    )

    assert request.state["previous_ai_message"] == "요즘 즐겨 하는 운동 있어요?"
    assert request.sentences == ("러닝은 진짜 싫어요.", "에어팟 맥스는 갖고 싶어요")
    assert set(request.questions) == {
        "none_answer",
        "stance_0",
        "material_0",
        "interest_category_0",
        "specificity_0",
        "explicit_0",
        "habitual_0",
        "evidence_0",
        "stance_1",
        "material_1",
        "interest_category_1",
        "specificity_1",
        "explicit_1",
        "deferral_1",
        "evidence_1",
    }
    assert request.questions["stance_0"]["instructions"]["candidate"] == "러닝"
    assert request.questions["evidence_1"]["criteria"] == {
        "s0": "러닝은 진짜 싫어요.",
        "s1": "에어팟 맥스는 갖고 싶어요",
    }


def test_judgment_maps_stance_and_kind_to_profile_fields() -> None:
    candidates = [
        Candidate("러닝", CandidateKind.ACTIVITY),
        Candidate("뜨개질", CandidateKind.ACTIVITY),
        Candidate("에어팟 맥스", CandidateKind.THING),
        Candidate("고소한 커피", CandidateKind.ATTRIBUTE, PreferenceAspect.SENSORY, "커피"),
        Candidate("트로트", CandidateKind.THING),
        Candidate("클라이밍", CandidateKind.ACTIVITY),
    ]
    sentences = ("러닝은 싫어요.", "뜨개질은 매일 해요.")
    answers = {
        "stance_0": choice("dislike"),
        "explicit_0": noul(0.97),
        "evidence_0": choice("s0"),
        "stance_1": choice("like"),
        "habitual_1": noul(0.98),
        "explicit_1": noul(0.9),
        "evidence_1": choice("s1"),
        "stance_2": choice("want"),
        "deferral_2": choice("price", 0.9),
        "stance_3": choice("like"),
        "explicit_3": noul(0.2),
        "stance_4": choice("other_person"),
        "stance_5": choice("like", 0.44),
        "none_answer": noul(0.1),
    }

    delta, judgments = judgment_delta(candidates, answers, sentences)

    by_value = {item.value: item for item in delta.items}
    assert set(by_value) == {"러닝", "뜨개질", "에어팟 맥스", "고소한 커피"}
    assert by_value["러닝"].field == TasteField.DISLIKES
    assert by_value["러닝"].evidence == "러닝은 싫어요."
    assert by_value["뜨개질"].field == TasteField.HOBBIES
    assert by_value["뜨개질"].evidence == "뜨개질은 매일 해요."
    assert by_value["에어팟 맥스"].field == TasteField.UNAFFORDABLE
    assert by_value["에어팟 맥스"].deferral_reason == DeferralReason.PRICE
    assert by_value["고소한 커피"].field == TasteField.PREFERENCES
    assert by_value["고소한 커피"].aspect == PreferenceAspect.SENSORY
    assert by_value["고소한 커피"].target == "커피"
    assert by_value["고소한 커피"].evidence_type == EvidenceType.INFERRED
    assert delta.none_answer is False
    assert [judgment.field for judgment in judgments][4:] == [None, None]


def test_wished_activity_becomes_low_confidence_interest() -> None:
    delta, _ = judgment_delta(
        [Candidate("도자기 공방", CandidateKind.ACTIVITY)],
        {"stance_0": choice("want", 0.99), "explicit_0": noul(0.9)},
        ("언젠가 도자기 배워 보고 싶어요",),
    )

    (item,) = delta.items
    assert item.field == TasteField.INTERESTS
    assert item.confidence == 0.5
    assert item.evidence_type == EvidenceType.INFERRED


def test_none_answer_comes_from_judgment() -> None:
    delta, _ = judgment_delta([], {"none_answer": noul(0.93)}, ("딱히 없어요",))

    assert delta.items == ()
    assert delta.none_answer is True


def test_reason_or_feeling_candidate_is_not_stored() -> None:
    delta, judgments = judgment_delta(
        [Candidate("숨차서", CandidateKind.THING)],
        {"stance_0": choice("not_a_subject", 0.9), "explicit_0": noul(0.9)},
        ("러닝은 숨차서 싫어요",),
    )

    assert delta.items == ()
    assert judgments[0].field is None


def test_attribute_candidate_is_asked_for_taste_axis() -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    request = build_judgment_request(
        state,
        "혼자 가는 게 좋아요",
        [Candidate("혼자 가는 캠핑", CandidateKind.ATTRIBUTE, target="캠핑")],
    )

    assert "taste_axis_0" in request.questions
    assert "interest_category_0" not in request.questions
    assert "social.size" in request.questions["taste_axis_0"]["criteria"]
    assert "none" in request.questions["taste_axis_0"]["criteria"]


def test_taste_axis_sets_taxonomy_path_and_aspect() -> None:
    delta, judgments = judgment_delta(
        [
            Candidate("혼자 가는 캠핑", CandidateKind.ATTRIBUTE, target="캠핑"),
            Candidate("무채색 장비", CandidateKind.ATTRIBUTE, PreferenceAspect.SENSORY, "캠핑"),
            Candidate("캠핑", CandidateKind.ACTIVITY),
        ],
        {
            "stance_0": choice("like"),
            "material_0": choice("taste"),
            "taste_axis_0": choice("social.size", 0.9),
            "stance_1": choice("like"),
            "material_1": choice("taste"),
            "taste_axis_1": choice("visual.color", 0.9),
            "stance_2": choice("like"),
            "material_2": choice("interest"),
            "interest_category_2": choice("outdoor", 0.95),
            "habitual_2": noul(0.9),
            "specificity_2": {"type": "score", "score": 1.6},
        },
        ("주말마다 혼자 캠핑 가요. 장비는 무채색으로 맞췄어요",),
    )

    by_value = {item.value: item for item in delta.items}
    assert by_value["혼자 가는 캠핑"].field == TasteField.PREFERENCES
    assert by_value["혼자 가는 캠핑"].taxonomy_path == ("취향", "사회", "인원")
    assert by_value["혼자 가는 캠핑"].aspect == PreferenceAspect.SITUATION
    # 축에서 정한 aspect가 추출 모델이 붙인 aspect보다 우선한다.
    assert by_value["무채색 장비"].aspect == PreferenceAspect.ATTRIBUTE
    assert by_value["캠핑"].field == TasteField.HOBBIES
    assert by_value["캠핑"].taxonomy_path == ("관심사", "아웃도어")
    assert judgments[0].material is not None


def test_not_material_is_dropped_even_when_liked() -> None:
    delta, judgments = judgment_delta(
        [Candidate("김치찌개", CandidateKind.THING), Candidate("텐트", CandidateKind.THING)],
        {
            "stance_0": choice("like"),
            "material_0": choice("not_material", 0.98),
            "stance_1": choice("owned"),
            "material_1": choice("not_material", 0.9),
        },
        ("점심에 김치찌개 먹었는데 맛있었어요. 텐트는 있어요",),
    )

    assert [(item.field, item.value) for item in delta.items] == [(TasteField.OWNED, "텐트")]
    assert judgments[0].field is None


def test_material_overrides_candidate_kind_for_taste() -> None:
    delta, _ = judgment_delta(
        [Candidate("조용한 카페", CandidateKind.THING)],
        {"stance_0": choice("like"), "material_0": choice("taste", 0.85)},
        ("조용한 카페가 좋아요",),
    )

    (item,) = delta.items
    assert item.field == TasteField.PREFERENCES


def test_broad_interest_gets_lower_confidence() -> None:
    delta, _ = judgment_delta(
        [
            Candidate("운동", CandidateKind.ACTIVITY),
            Candidate("데드리프트", CandidateKind.ACTIVITY),
        ],
        {
            "stance_0": choice("like", 0.95),
            "specificity_0": {"type": "score", "score": 0.6},
            "stance_1": choice("like", 0.95),
            "specificity_1": {"type": "score", "score": 1.8},
        },
        ("운동 좋아해요. 데드리프트 위주로요",),
    )

    by_value = {item.value: item for item in delta.items}
    assert by_value["운동"].confidence == 0.55
    assert by_value["데드리프트"].confidence == 0.95


def test_attribute_with_taste_axis_stays_taste_even_if_material_says_interest() -> None:
    state = ConversationState(user_id=1, conversation_room_id=1)
    candidate = Candidate("혼자 가는 캠핑", CandidateKind.ATTRIBUTE, target="캠핑")
    request = build_judgment_request(state, "혼자 캠핑 가요", [candidate])
    delta, _ = judgment_delta(
        [candidate],
        {
            "stance_0": choice("like", 0.74),
            "material_0": choice("interest", 0.8),
            "taste_axis_0": choice("social.size", 0.9),
        },
        request.sentences,
    )

    assert set(request.questions["stance_0"]["criteria"]) == {
        "like",
        "dislike",
        "used_to_like",
        "mention_only",
        "other_person",
    }
    (item,) = delta.items
    assert item.field == TasteField.PREFERENCES
    assert item.taxonomy_path == ("취향", "사회", "인원")


def _profile_with(*items: tuple[TasteField, str, str], turn: int = 1) -> ProfileState:
    profile = ProfileState()
    for field, value, evidence in items:
        profile = (
            ProfileMerger()
            .merge(
                profile,
                ExtractionDelta(
                    items=(
                        ExtractedItem(
                            field=field,
                            value=value,
                            confidence=0.9,
                            evidence=evidence,
                            evidence_type=EvidenceType.EXPLICIT,
                        ),
                    )
                ),
                utterance=evidence,
                now=NOW,
                source_turn=turn,
            )
            .profile
        )
    return profile


def test_correction_targets_prefer_mentioned_then_recent_items() -> None:
    profile = _profile_with(
        (TasteField.HOBBIES, "캠핑", "캠핑 자주 가요"),
        (TasteField.INTERESTS, "재즈", "재즈 들어요"),
    )
    old = _profile_with((TasteField.INTERESTS, "요리", "요리 좋아해요"), turn=1)
    profile.signals.extend(old.signals)
    for signal in profile.signals:
        if signal.value == "재즈":
            signal.source_turn = 5
    state = ConversationState(user_id=1, conversation_room_id=1, profile=profile)
    state.turn_count = 5

    targets = correction_targets(state, "아 캠핑은 사실 그냥 해본 말이에요")

    assert [signal.value for signal in targets] == ["캠핑", "재즈"]


def test_correction_questions_are_added_only_with_targets() -> None:
    profile = _profile_with((TasteField.HOBBIES, "캠핑", "캠핑 자주 가요"))
    state = ConversationState(user_id=1, conversation_room_id=1, profile=profile)

    without = build_judgment_request(state, "등산 좋아해요", [])
    with_targets = build_judgment_request(state, "등산 좋아해요", [], profile.stored_signals())

    assert "correction_intent" not in without.questions
    assert {"correction_intent", "retract_0"} <= set(with_targets.questions)
    assert with_targets.state["profile_items"] == {
        "p0": "캠핑 (an activity the user does regularly)"
    }


def test_retraction_needs_both_turn_intent_and_item_judgment() -> None:
    profile = _profile_with(
        (TasteField.HOBBIES, "캠핑", "캠핑 자주 가요"),
        (TasteField.DISLIKES, "견과류", "견과류 싫어요"),
    )
    targets = profile.stored_signals()

    def drops(intent: float, camping: float, nuts: float) -> set[str]:
        delta, _ = judgment_delta(
            [],
            {
                "correction_intent": noul(intent),
                "retract_0": noul(camping),
                "retract_1": noul(nuts),
            },
            ("캠핑은 그냥 해본 말이에요",),
            targets=targets,
        )
        return {drop.value for drop in delta.drop}

    assert drops(0.9, 0.9, 0.2) == {"캠핑"}
    assert drops(0.3, 0.9, 0.2) == set()
    # 싫은 것은 안전 정보라 0.85를 넘어야 내린다.
    assert drops(0.9, 0.1, 0.8) == set()
    assert drops(0.9, 0.1, 0.9) == {"견과류"}


def test_used_to_like_drops_the_stored_positive_item() -> None:
    profile = _profile_with((TasteField.HOBBIES, "뜨개질", "뜨개질 매일 해요"))

    delta, _ = judgment_delta(
        [Candidate("뜨개질", CandidateKind.ACTIVITY)],
        {"stance_0": choice("used_to_like", 0.9)},
        ("뜨개질 요즘은 안 해요",),
        targets=profile.stored_signals(),
    )

    assert delta.items == ()
    assert [(drop.field, drop.value) for drop in delta.drop] == [(TasteField.HOBBIES, "뜨개질")]
