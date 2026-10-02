import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.application.conversation_prompts import (
    BROADEN_INSTRUCTION,
    build_extraction_messages,
    build_friend_summary_messages,
    build_reply_messages,
)
from app.application.conversation_service import ConversationService, parse_extraction
from app.application.ports.model_gateway import Message
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    ConversationTurn,
    CoverageStatus,
    GoalArea,
    SessionStatus,
)
from app.domain.conversation.policy import decide_goal
from app.domain.profile.models import (
    EvidenceType,
    IntentType,
    LinkRole,
    PreferenceAspect,
    ProfileSignal,
    SignalStatus,
    TasteField,
    Visibility,
)


class FakeModelGateway:
    def __init__(
        self,
        structured_results: list[Mapping[str, Any] | Exception],
    ) -> None:
        self.structured_results = structured_results
        self.calls = 0

    async def complete(self, messages: Sequence[Message], *, model: str) -> str:
        raise NotImplementedError

    def stream(self, messages: Sequence[Message], *, model: str) -> AsyncIterator[str]:
        raise NotImplementedError

    async def structured(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = self.structured_results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def taste_signal(value: str, path: tuple[str, ...]) -> ProfileSignal:
    signal = profile_signal(TasteField.PREFERENCES, value)
    signal.link_role = LinkRole.WEIGHT
    signal.taxonomy_path = path
    return signal


def profile_signal(field: TasteField, value: str) -> ProfileSignal:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    return ProfileSignal(
        field=field,
        value=value,
        normalized_value=value,
        confidence=0.9,
        link_role=LinkRole.QUERY,
        visibility=Visibility.FRIENDS,
        intent_type=IntentType.BOTH,
        deferral_reason=None,
        evidence=value,
        evidence_type=EvidenceType.EXPLICIT,
        first_seen_at=now,
        updated_at=now,
        status=SignalStatus.ACTIVE,
    )


def test_prepare_session_uses_room_as_session_id_and_records_safe_opening() -> None:
    gateway = FakeModelGateway([])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    now = datetime(2026, 9, 18, tzinfo=UTC)

    prepared = service.prepare_session(
        user_id=1,
        conversation_room_id=101,
        now=now,
    )
    greeting = service.record_opening(
        prepared.state,
        "반가워요! 요즘 어떻게 지내세요? 두 번째 질문? <<<STATE>>>비밀",
        now=now,
    )

    assert prepared.state.session_id == 101
    assert len(prepared.messages) == 2
    assert greeting == "반가워요! 요즘 어떻게 지내세요?"
    assert prepared.state.history[0].content == greeting
    assert prepared.state.history[0].created_at == now
    assert prepared.state.turn_count == 0


def test_complete_turn_extracts_merges_and_updates_state() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    {
                        "field": "hobbies",
                        "value": "핸드드립",
                        "confidence": 0.9,
                        "evidence": "핸드드립을 자주 해요",
                        "evidenceType": "explicit",
                        "intentType": "both",
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [],
                "noneAnswer": False,
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)
    prepared = service.prepare_turn(state, "핸드드립을 자주 해요")

    assert state.goal_attempts == {}

    completed_at = datetime(2026, 9, 18, tzinfo=UTC)
    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="핸드드립을 자주 해요",
            raw_reply="주로 어떤 원두를 사용하세요? 두 번째 질문? <<<STATE>>>비밀",
            goal=prepared.decision.goal,
            now=completed_at,
        )
    )

    assert prepared.decision.goal == ConversationGoal.INTEREST
    assert completed.reply == "주로 어떤 원두를 사용하세요?"
    assert completed.state.turn_count == 1
    assert completed.state.profile.active_signals()[0].value == "핸드드립"
    assert [turn.created_at for turn in completed.state.history] == [
        completed_at,
        completed_at,
    ]
    assert completed.extraction_failed is False
    assert completed.state.goal_attempts[ConversationGoal.INTEREST.value] == 1


def test_invalid_extraction_retries_then_uses_empty_delta() -> None:
    gateway = FakeModelGateway([["not an object"], ["not an object"], ["not an object"]])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="잘 모르겠어요",
            raw_reply="평소 쉬는 날에는 어떻게 보내세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert gateway.calls == 3
    assert completed.extraction_failed is True
    assert completed.state.profile.active_signals() == []


def test_model_failure_during_extraction_does_not_break_completed_chat_turn() -> None:
    gateway = FakeModelGateway(
        [RuntimeError("model unavailable"), RuntimeError("model unavailable")]
    )
    service = ConversationService(
        model_gateway=gateway,
        extraction_model="extractor",
        extraction_retries=1,
    )
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="잘 모르겠어요",
            raw_reply="평소 쉬는 날에는 어떻게 보내세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert gateway.calls == 2
    assert completed.extraction_failed is True
    assert completed.state.turn_count == 1


def test_friend_summary_prompt_never_contains_sensitive_profile_values() -> None:
    # Projection behavior itself is covered in test_profile_merger; this ensures the prompt uses it.
    state = ConversationState(user_id=1, conversation_room_id=101)

    messages = build_friend_summary_messages(state.profile)

    assert len(messages) == 2
    assert "경제 사정" in messages[0]["content"]


def test_item_outside_goal_fields_leaves_goal_unresolved() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    {
                        "field": "preferences",
                        "value": "가벼운 것",
                        "confidence": 0.9,
                        "evidence": "가벼운 게 좋아요",
                        "evidenceType": "explicit",
                        "intentType": None,
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [],
                "noneAnswer": False,
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="가벼운 게 좋아요",
            raw_reply="요즘 실제로 자주 하는 일은 무엇인가요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.state.goal_coverage[GoalArea.INTEREST] == CoverageStatus.UNRESOLVED


def test_extraction_that_completes_readiness_closes_before_eight_turns() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    {
                        "field": "interests",
                        "value": "사진",
                        "confidence": 0.9,
                        "evidence": "사진도 좋아해요",
                        "evidenceType": "explicit",
                        "intentType": "both",
                        "deferralReason": None,
                    }
                ],
                "axes": [],
                "drop": [],
                "noneAnswer": False,
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.profile.signals.extend(
        [
            profile_signal(TasteField.INTERESTS, "캠핑"),
            profile_signal(TasteField.HOBBIES, "핸드드립"),
            profile_signal(TasteField.INTERESTS, "독서"),
            profile_signal(TasteField.HOBBIES, "러닝"),
            taste_signal("혼자 가는 캠핑", ("취향", "사회", "인원")),
            taste_signal("조용한 카페", ("취향", "분위기", "공간")),
        ]
    )
    state.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE
    state.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.CONFIRMED_NONE

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="사진도 좋아해요",
            raw_reply="사진은 주로 언제 찍으세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.readiness.sufficient is True
    assert completed.state.turn_count == 1
    assert completed.next_decision.completion_reason == CompletionReason.SUFFICIENT
    assert completed.state.status == SessionStatus.INPUT_LOCKED
    assert completed.state.completion_reason == CompletionReason.SUFFICIENT
    assert decide_goal(completed.state, "").completion_reason == CompletionReason.SUFFICIENT


def _raw_item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "field": "interests",
        "value": "캠핑",
        "confidence": 0.8,
        "evidence": "캠핑 다녀왔어요",
        "evidenceType": "explicit",
        "intentType": None,
        "deferralReason": None,
    }
    item.update(overrides)
    return item


def test_parse_extraction_keeps_valid_items_when_others_are_invalid() -> None:
    delta, dropped = parse_extraction(
        {
            "items": [
                _raw_item(value="이 값은 스무 글자를 훌쩍 넘어가는 아주 긴 값입니다"),
                _raw_item(field="unknown"),
                _raw_item(value="러닝", evidence="뛰어요"),
            ],
        }
    )

    assert [item.value for item in delta.items] == ["러닝"]
    assert len(dropped) == 2
    assert delta.none_answer is False


def test_parse_extraction_repairs_evidence_and_deferral_and_caps_items() -> None:
    delta, _ = parse_extraction(
        {
            "items": [
                _raw_item(evidence="가" * 50),
                _raw_item(field="unaffordable", value="카메라"),
                _raw_item(value="요리", deferralReason="price"),
                _raw_item(value="넷째"),
                _raw_item(value="다섯째"),
                _raw_item(value="여섯째"),
            ],
            "noneAnswer": True,
        }
    )

    assert [item.value for item in delta.items] == ["캠핑", "카메라", "요리", "넷째", "다섯째"]
    assert len(delta.items[0].evidence) == 40
    assert delta.items[1].field == TasteField.WANTS
    assert delta.items[2].deferral_reason is None
    assert delta.none_answer is True


def test_none_answer_confirms_gear_goal_as_none() -> None:
    gateway = FakeModelGateway([{"items": [], "noneAnswer": True}])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="딱히 챙기는 건 없어요",
            raw_reply="그렇군요. 요즘은 뭐 하면서 쉬어요?",
            goal=ConversationGoal.GEAR,
        )
    )

    assert completed.state.goal_coverage[GoalArea.GEAR] == CoverageStatus.CONFIRMED_NONE


def test_dislike_found_on_interest_turn_marks_exclusion_found() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    _raw_item(field="dislikes", value="향 강한 것", evidence="향 강한 건 싫어요")
                ],
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="향 강한 건 싫어요",
            raw_reply="그럼 요즘 자주 쓰는 건 뭐예요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.state.goal_coverage[GoalArea.EXCLUSION] == CoverageStatus.FOUND
    assert completed.state.goal_coverage[GoalArea.INTEREST] == CoverageStatus.UNRESOLVED


def test_extraction_messages_include_numbered_recent_dialogue() -> None:
    now = datetime(2026, 9, 18, tzinfo=UTC)
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.history.extend(
        [
            ConversationTurn(role="assistant", content="요즘 어떻게 지내세요?", created_at=now),
            ConversationTurn(role="user", content="퇴근하고 요리해요", created_at=now),
            ConversationTurn(role="assistant", content="요리는 언제 제일 좋아요?", created_at=now),
        ]
    )

    content = build_extraction_messages(state, "그 시간이 제일 좋아요")[1]["content"]

    assert "사용자(1턴): 퇴근하고 요리해요" in content
    assert "AI: 요리는 언제 제일 좋아요?" in content
    assert "[현재 목표]" not in content


def test_opening_prompt_picks_topic_and_local_moment() -> None:
    import random

    from app.application.conversation_prompts import OPENING_TOPICS, build_opening_messages

    # 2026-09-29 11:00 UTC는 서울 기준 화요일 20시다.
    messages = build_opening_messages(
        now=datetime(2026, 9, 29, 11, 0, tzinfo=UTC),
        rng=random.Random(0),
    )

    content = messages[1]["content"]
    assert any(topic in content for topic in OPENING_TOPICS)
    assert "화요일 저녁, 가을" in content


def test_closing_reply_replaces_trailing_question_with_closing_message() -> None:
    from app.domain.conversation.guards import CLOSING_MESSAGE, closing_reply

    assert closing_reply("그런 주도 있죠. 내일 뭐 해요?") == f"그런 주도 있죠. {CLOSING_MESSAGE}"
    assert closing_reply("부모님 댁 가면 뭐 해요?") == CLOSING_MESSAGE


def test_turn_that_completes_readiness_ends_with_closing_message() -> None:
    from app.domain.conversation.guards import CLOSING_MESSAGE

    gateway = FakeModelGateway([{"items": [_raw_item(value="사진", evidence="사진도 좋아해요")]}])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.profile.signals.extend(
        [
            profile_signal(TasteField.INTERESTS, "캠핑"),
            profile_signal(TasteField.HOBBIES, "핸드드립"),
            profile_signal(TasteField.INTERESTS, "독서"),
            profile_signal(TasteField.HOBBIES, "러닝"),
            taste_signal("혼자 가는 캠핑", ("취향", "사회", "인원")),
            taste_signal("조용한 카페", ("취향", "분위기", "공간")),
        ]
    )
    state.goal_coverage[GoalArea.GEAR] = CoverageStatus.CONFIRMED_NONE
    state.goal_coverage[GoalArea.EXCLUSION] = CoverageStatus.CONFIRMED_NONE

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="사진도 좋아해요",
            raw_reply="사진 좋죠. 주로 뭘 찍어요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.reply == f"사진 좋죠. {CLOSING_MESSAGE}"
    assert completed.state.history[-1].content == completed.reply
    assert completed.state.status == SessionStatus.INPUT_LOCKED


def test_last_turn_at_max_turns_ends_with_closing_message() -> None:
    from app.domain.conversation.guards import CLOSING_MESSAGE
    from app.domain.conversation.policy import MAX_TURNS

    gateway = FakeModelGateway([{"items": []}])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101, turn_count=MAX_TURNS - 1)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="주말엔 그냥 쉬어요",
            raw_reply="쉬는 것도 중요하죠. 쉴 때는 주로 뭐 해요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.state.turn_count == MAX_TURNS
    assert completed.reply == f"쉬는 것도 중요하죠. {CLOSING_MESSAGE}"
    assert completed.state.history[-1].content == completed.reply
    assert completed.state.status == SessionStatus.INPUT_LOCKED
    assert completed.state.completion_reason == CompletionReason.MAX_CYCLES


def test_reflect_turn_at_max_turns_still_locks_with_closing_message() -> None:
    from app.domain.conversation.guards import CLOSING_MESSAGE
    from app.domain.conversation.models import ConversationStyle
    from app.domain.conversation.policy import MAX_TURNS

    gateway = FakeModelGateway([{"items": []}])
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(
        user_id=1,
        conversation_room_id=101,
        turn_count=MAX_TURNS - 1,
        conversation_style=ConversationStyle.REFLECTIVE,
    )

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="네 그런 것 같아요",
            raw_reply="혼자만의 시간을 좋아하시는 거죠?",
            goal=ConversationGoal.REFLECT,
        )
    )

    assert completed.reply == CLOSING_MESSAGE
    assert completed.state.status == SessionStatus.INPUT_LOCKED


def test_summary_prompt_sends_ranked_friend_clues_without_private_fields() -> None:
    import json

    state = ConversationState(user_id=1, conversation_room_id=101)
    state.profile.signals.extend(
        [
            profile_signal(TasteField.INTERESTS, "조용한 카페"),
            profile_signal(TasteField.HOBBIES, "핸드드립"),
            profile_signal(TasteField.DISLIKES, "강한 향"),
        ]
    )

    content = json.loads(build_friend_summary_messages(state.profile)[1]["content"])

    values = [item["value"] for group in content["interests"] for item in group["items"]]
    assert set(values) == {"조용한 카페", "핸드드립"}
    assert content["tastes"] == []


def test_summary_includes_stored_tastes_beyond_exposure_limit_grouped_by_axis() -> None:
    import json

    from app.domain.profile.models import SignalStatus

    state = ConversationState(user_id=1, conversation_room_id=101)
    alone_camping = taste_signal("혼자 가는 캠핑", ("취향", "사회", "인원"))
    alone_camping.target = "캠핑"
    alone_movie = taste_signal("혼자 보는 영화", ("취향", "사회", "인원"))
    alone_movie.target = "영화"
    alone_movie.status = SignalStatus.INACTIVE
    quiet = taste_signal("사람 없는 계곡", ("취향", "장소", "밀집도"))
    replaced = taste_signal("붐비는 캠핑장", ("취향", "장소", "밀집도"))
    replaced.status = SignalStatus.SUPERSEDED
    state.profile.signals.extend([quiet, alone_camping, alone_movie, replaced])

    content = json.loads(build_friend_summary_messages(state.profile)[1]["content"])

    # 서로 다른 관심사에 걸쳐 나온 축이 앞에 오고, 활성 한도 밖 항목도 들어간다.
    assert content["tastes"][0]["axis"] == "사회/인원"
    assert [item["value"] for item in content["tastes"][0]["items"]] == [
        "혼자 가는 캠핑",
        "혼자 보는 영화",
    ]
    assert content["tastes"][1] == {
        "axis": "장소/밀집도",
        "items": [
            {
                "kind": "좋아하는 방식과 스타일",
                "value": "사람 없는 계곡",
                "context": "사람 없는 계곡",
            }
        ],
    }


def test_parse_extraction_keeps_aspect_and_target_only_on_preferences() -> None:
    delta, dropped = parse_extraction(
        {
            "items": [
                _raw_item(
                    field="preferences",
                    value="고소한 커피",
                    evidence="고소한 게 좋더라고요",
                    aspect="sensory",
                    target="커피",
                ),
                _raw_item(value="커피", aspect="sensory", target="커피"),
                _raw_item(field="preferences", value="잔잔한 음악", aspect="unknown"),
            ],
        }
    )

    preference, interest = delta.items
    assert preference.aspect == PreferenceAspect.SENSORY
    assert preference.target == "커피"
    assert interest.aspect is None and interest.target is None
    assert len(dropped) == 1


def test_preference_aspect_and_target_survive_merge_and_session_round_trip() -> None:
    gateway = FakeModelGateway(
        [
            {
                "items": [
                    _raw_item(
                        field="preferences",
                        value="조용한 구석 자리",
                        evidence="구석 자리가 조용해서",
                        evidenceType="inferred",
                        confidence=0.6,
                        aspect="situation",
                        target="카페",
                    )
                ]
            }
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="구석 자리가 조용해서 두 시간이나 있었어요",
            raw_reply="그 카페는 자주 가요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    restored = ConversationState.from_dict(completed.state.to_dict())
    [signal] = restored.profile.active_signals()
    assert signal.aspect == PreferenceAspect.SITUATION
    assert signal.target == "카페"


def _extraction(*items: dict[str, Any]) -> dict[str, Any]:
    return {"items": list(items), "axes": [], "drop": [], "noneAnswer": False}


def test_turns_without_new_interest_broaden_the_next_question() -> None:
    hiking = {
        "field": "hobbies",
        "value": "등산",
        "confidence": 0.9,
        "evidence": "주말마다 산에 가요",
        "evidenceType": "explicit",
        "intentType": "both",
    }
    quiet_trail = {
        "field": "preferences",
        "value": "사람 적은 등산로",
        "confidence": 0.8,
        "evidence": "사람 없는 길이 좋아요",
        "evidenceType": "explicit",
        "intentType": "both",
        "aspect": "situation",
        "target": "등산",
    }
    hiking_again = {**hiking, "evidence": "산은 매주 가요"}
    gateway = FakeModelGateway(
        [_extraction(hiking), _extraction(quiet_trail), _extraction(hiking_again)]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)
    utterances = ["주말마다 산에 가요", "사람 없는 길이 좋아요", "산은 매주 가요"]

    counts = []
    for utterance in utterances:
        asyncio.run(
            service.complete_turn(
                state,
                utterance=utterance,
                raw_reply="그렇군요?",
                goal=ConversationGoal.INTEREST,
            )
        )
        counts.append(state.turns_since_new_interest)

    assert counts == [0, 1, 2]
    interest_prompt = build_reply_messages(state, ConversationGoal.INTEREST, "정상에서 쉬어요")
    assert BROADEN_INSTRUCTION in interest_prompt[-2]["content"]
    gear_prompt = build_reply_messages(state, ConversationGoal.GEAR, "정상에서 쉬어요")
    assert BROADEN_INSTRUCTION not in gear_prompt[-2]["content"]
    assert ConversationState.from_dict(state.to_dict()).turns_since_new_interest == 2


def test_new_interest_resets_topic_counter() -> None:
    gateway = FakeModelGateway(
        [
            _extraction(
                {
                    "field": "interests",
                    "value": "평양냉면",
                    "confidence": 0.8,
                    "evidence": "산 다녀오면 냉면 먹어요",
                    "evidenceType": "explicit",
                    "intentType": "both",
                }
            )
        ]
    )
    service = ConversationService(model_gateway=gateway, extraction_model="extractor")
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.profile.signals.append(profile_signal(TasteField.HOBBIES, "등산"))
    state.turns_since_new_interest = 2

    asyncio.run(
        service.complete_turn(
            state,
            utterance="산 다녀오면 냉면 먹어요",
            raw_reply="어디 냉면 좋아해요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert state.turns_since_new_interest == 0
    prompt = build_reply_messages(state, ConversationGoal.INTEREST, "을밀대요")
    assert BROADEN_INSTRUCTION not in prompt[-2]["content"]


class FakeJudgmentGateway:
    def __init__(self, answers: Mapping[str, Any]) -> None:
        self.answers = answers
        self.questions: Mapping[str, Any] = {}

    async def decide(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        model: str,
    ) -> Mapping[str, Any]:
        self.questions = questions
        return {"model": model, "answers": self.answers}


def test_judgment_path_stores_disliked_subject_only_as_dislike() -> None:
    gateway = FakeModelGateway(
        [
            {
                "candidates": [
                    {"subject": "러닝", "kind": "activity"},
                    {"subject": "걷기", "kind": "activity"},
                ]
            }
        ]
    )
    judge = FakeJudgmentGateway(
        {
            "stance_0": {"type": "choice", "choice": "dislike", "confidence": 1.0},
            "explicit_0": {"type": "noul", "noul": 0.95},
            "evidence_0": {"type": "choice", "choice": "s0", "confidence": 0.9},
            "stance_1": {"type": "choice", "choice": "like", "confidence": 0.9},
            "habitual_1": {"type": "noul", "noul": 0.1},
            "explicit_1": {"type": "noul", "noul": 0.8},
            "evidence_1": {"type": "choice", "choice": "s2", "confidence": 0.9},
        }
    )
    service = ConversationService(
        model_gateway=gateway,
        extraction_model="extractor",
        judgment_gateway=judge,
        judgment_model="~typesafe/jev-latest",
    )
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="러닝은 진짜 싫어요. 숨차서요. 그냥 걷는 건 괜찮아요",
            raw_reply="걷는 건 주로 언제 하세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    active = {(signal.field, signal.value) for signal in completed.state.profile.active_signals()}
    assert active == {(TasteField.DISLIKES, "러닝"), (TasteField.INTERESTS, "걷기")}
    assert completed.extraction_failed is False
    assert "habitual_0" in judge.questions


def test_judgment_failure_marks_extraction_failed_without_breaking_turn() -> None:
    class BrokenJudgmentGateway:
        async def decide(self, state, questions, *, model):
            raise RuntimeError("down")

    gateway = FakeModelGateway([{"candidates": [{"subject": "러닝", "kind": "activity"}]}])
    service = ConversationService(
        model_gateway=gateway,
        extraction_model="extractor",
        judgment_gateway=BrokenJudgmentGateway(),
        judgment_model="~typesafe/jev-latest",
    )
    state = ConversationState(user_id=1, conversation_room_id=101)

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="러닝 좋아해요",
            raw_reply="언제 뛰세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    assert completed.extraction_failed is True
    assert completed.state.profile.active_signals() == []
    assert completed.reply == "언제 뛰세요?"


def test_judgment_path_replaces_corrected_item_with_the_new_one() -> None:
    gateway = FakeModelGateway([{"candidates": [{"subject": "등산", "kind": "activity"}]}])
    judge = FakeJudgmentGateway(
        {
            "stance_0": {"type": "choice", "choice": "like", "confidence": 0.95},
            "habitual_0": {"type": "noul", "noul": 0.8},
            "explicit_0": {"type": "noul", "noul": 0.9},
            "correction_intent": {"type": "noul", "noul": 0.93},
            "retract_0": {"type": "noul", "noul": 0.9},
        }
    )
    service = ConversationService(
        model_gateway=gateway,
        extraction_model="extractor",
        judgment_gateway=judge,
        judgment_model="~typesafe/jev-latest",
    )
    state = ConversationState(user_id=1, conversation_room_id=101)
    state.profile.signals.append(profile_signal(TasteField.HOBBIES, "캠핑"))
    state.turn_count = 1

    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance="아 캠핑이 아니라 등산이요",
            raw_reply="등산은 주로 어디로 가세요?",
            goal=ConversationGoal.INTEREST,
        )
    )

    active = {(signal.field, signal.value) for signal in completed.state.profile.active_signals()}
    assert active == {(TasteField.HOBBIES, "등산")}
    assert "retract_0" in judge.questions
