import asyncio
import json
from datetime import UTC, datetime

import pytest

from app.api.schemas.memory import ConversationMemory
from app.application.conversation_prompts import build_reply_messages
from app.application.conversation_service import ConversationService, parse_extraction
from app.application.profile_summary_prompts import build_friend_summary_messages
from app.application.recommendation_service import recommendation_inputs
from app.application.taste_judgment import Candidate, CandidateKind, judgment_delta
from app.domain.conversation.models import (
    ConversationGoal,
    ConversationState,
)
from app.domain.conversation.policy import recommendation_readiness
from app.domain.profile.memory import merge_memory
from app.domain.profile.merger import ProfileMerger
from app.domain.profile.models import (
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
    InsightKind,
    PreferenceAspect,
    ProfileSignal,
    ProfileState,
    TasteField,
)

NOW = datetime(2026, 11, 4, tzinfo=UTC)
UTTERANCE = "큰 결정을 할 때는 제가 납득해야 골라요. 예전에 남의 말만 듣고 골랐다가 후회해서요."


def value_payload(**changes):
    return {
        "field": "preferences",
        "value": "스스로 납득하는 선택",
        "confidence": 0.9,
        "evidence": "제가 납득해야 골라요",
        "evidenceType": "explicit",
        "insightKind": "value",
        "context": "큰 결정을 할 때",
        "reason": "후회해서요",
        "origin": "예전에 남의 말만 듣고 골랐다가 후회해서요",
        **changes,
    }


def merge_item(payload=None, utterance=UTTERANCE):
    delta, dropped = parse_extraction({"items": [payload or value_payload()]})
    assert not dropped
    return ProfileMerger().merge(
        ProfileState(), delta, utterance=utterance, now=NOW, source_turn=2, conversation_id=101
    )


def ready_state():
    state = ConversationState(user_id=1, conversation_room_id=101)
    values = ("캠핑", "독서", "러닝", "재즈", "사진", "뜨개질")
    items = [ExtractedItem(TasteField.INTERESTS, v, 0.9, v, EvidenceType.EXPLICIT) for v in values]
    items.extend(
        ExtractedItem(
            TasteField.PREFERENCES,
            v,
            0.9,
            v,
            EvidenceType.EXPLICIT,
            aspect=PreferenceAspect.SITUATION,
            taxonomy_path=("취향", "상황", v),
        )
        for v in ("조용한 카페", "혼자 하는 독서")
    )
    state.profile = (
        ProfileMerger()
        .merge(
            ProfileState(),
            ExtractionDelta(items=tuple(items)),
            utterance=" ".join(item.value for item in items),
            now=NOW,
            source_turn=1,
        )
        .profile
    )
    state.turn_count = 1
    assert recommendation_readiness(state).sufficient
    return state


class Extractor:
    def __init__(self, body):
        self.body = body
        self.calls = 0

    async def structured(self, *args, **kwargs):
        self.calls += 1
        return self.body


class Judge:
    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    async def decide(self, *args, **kwargs):
        self.calls += 1
        return {"answers": self.answers}


def choice(value):
    return {"choice": value, "confidence": 0.95}


def noul(value):
    return {"noul": value}


@pytest.mark.parametrize("kind", [InsightKind.VALUE, InsightKind.LIFE_ATTITUDE])
def test_explicit_principle_keeps_its_context_and_origin_without_requiring_an_activity(kind):
    merged = merge_item(value_payload(insightKind=kind.value))
    (signal,) = merged.accepted
    assert signal.insight_kind == kind
    assert signal.target is None
    assert signal.reason.text == "후회해서요"
    assert signal.origin.text == "예전에 남의 말만 듣고 골랐다가 후회해서요"
    assert signal.context.source_turn == 2
    assert signal.context.conversation_id == 101
    assert ProfileSignal.from_dict(signal.to_dict()) == signal


def test_context_from_ai_or_an_invented_quote_is_discarded_without_losing_the_preference():
    (signal,) = merge_item(
        value_payload(reason="독립적인 성격이라서", origin="어릴 때부터 그랬어요")
    ).accepted
    assert signal.reason is None
    assert signal.origin is None
    assert signal.context is not None


def test_inferred_worldview_is_not_saved_as_a_personal_value():
    merged = merge_item(value_payload(evidenceType="inferred"))
    assert not merged.accepted
    assert merged.rejected_values == ("스스로 납득하는 선택",)


def test_new_reason_updates_an_existing_preference_without_inflating_mention_count():
    first = ProfileMerger().merge(
        ProfileState(),
        ExtractionDelta(
            items=(
                ExtractedItem(
                    TasteField.PREFERENCES,
                    "조용한 자리",
                    0.9,
                    "조용한 자리가 좋아요",
                    EvidenceType.EXPLICIT,
                ),
            )
        ),
        utterance="조용한 자리가 좋아요",
        now=NOW,
        source_turn=1,
    )
    second = ProfileMerger().merge(
        first.profile,
        ExtractionDelta(
            items=(
                ExtractedItem(
                    TasteField.PREFERENCES,
                    "조용한 자리",
                    0.9,
                    "조용한 자리가 좋아요",
                    EvidenceType.EXPLICIT,
                    reason="사람들이 오가면 책을 못 읽어서요",
                ),
            )
        ),
        utterance="사람들이 오가면 책을 못 읽어서요",
        now=NOW,
        source_turn=2,
        context_utterances=[(1, "조용한 자리가 좋아요")],
        conversation_id=101,
    )
    (signal,) = second.profile.signals
    assert signal.mention_count == 1
    assert signal.reason.source_turn == 2


def test_memory_wire_round_trip_keeps_quotes_and_accepts_legacy_signals():
    memory = merge_memory(None, merge_item().profile, user_id=1, now=NOW)
    wire = ConversationMemory.from_domain(memory).model_dump(mode="json", by_alias=True)
    restored = ConversationMemory.model_validate(wire).to_domain()
    (signal,) = restored.signals
    assert signal.insight_kind == InsightKind.VALUE
    assert signal.origin.conversation_id == 101
    assert signal.origin.source_turn == 2
    old = dict(wire)
    old["signals"] = [dict(wire["signals"][0])]
    for key in ("insightKind", "context", "reason", "origin"):
        old["signals"][0].pop(key)
    assert (
        ConversationMemory.model_validate(old).to_domain().signals[0].insight_kind
        == InsightKind.PREFERENCE
    )


def test_judgment_verifies_the_relation_and_preserves_the_existing_parallel_call_count():
    extractor = Extractor(
        {
            "candidates": [
                {
                    "subject": "스스로 납득하는 선택",
                    "kind": "attribute",
                    "insight_kind": "value",
                    "reason": "후회해서요",
                    "origin": "예전에 남의 말만 듣고 골랐다가 후회해서요",
                }
            ],
            "conversation": {
                "focus": "스스로 선택하는 기준",
                "direction": "developing",
                "evidence": "제가 납득해야 골라요",
            },
        }
    )
    judge = Judge(
        {
            "stance_0": choice("like"),
            "material_0": choice("taste"),
            "explicit_0": noul(0.95),
            "meaning_0": noul(0.95),
            "reason_0": noul(0.1),
            "origin_0": noul(0.95),
        }
    )
    service = ConversationService(
        model_gateway=extractor,
        extraction_model="extractor",
        judgment_gateway=judge,
        judgment_model="judge",
    )
    state = ConversationState(user_id=1, conversation_room_id=101)
    completed = asyncio.run(
        service.complete_turn(
            state,
            utterance=UTTERANCE,
            raw_reply="스스로 납득하는 게 중요하군요.",
            goal=ConversationGoal.CHAT,
        )
    )
    (signal,) = completed.merge_result.accepted
    assert signal.insight_kind == InsightKind.VALUE
    assert signal.reason is None
    assert signal.origin is not None
    assert extractor.calls == judge.calls == 1


def test_judge_does_not_promote_a_hypothesis_into_a_value():
    delta, _ = judgment_delta(
        [Candidate("독립적인 선택", CandidateKind.ATTRIBUTE, insight_kind=InsightKind.VALUE)],
        {
            "stance_0": choice("like"),
            "material_0": choice("taste"),
            "explicit_0": noul(0.1),
            "meaning_0": noul(0.95),
        },
        ["혼자 카페에 갔어요"],
    )
    assert not delta.items


def test_summary_receives_values_but_not_detailed_context_or_source_quotes():
    profile = merge_item().profile
    body = json.loads(build_friend_summary_messages(profile)[1]["content"])
    assert body["tastes"][0]["items"][0]["kind"] == "직접 말한 가치 기준"
    assert body["tastes"][0]["items"][0]["value"] == "스스로 납득하는 선택"
    assert "후회해서요" not in json.dumps(body, ensure_ascii=False)
    assert "큰 결정을 할 때" not in json.dumps(body, ensure_ascii=False)
    messages = build_reply_messages(
        ConversationState(user_id=1, conversation_room_id=101, profile=profile),
        ConversationGoal.CHAT,
        "아까 말한 기준이요",
    )
    assert any("후회해서요" in m["content"] for m in messages)


def test_values_are_not_used_as_product_features_or_catalog_exclusions():
    payload = {
        "profile": {
            "preferences": [
                {
                    "value": "스스로 납득하는 선택",
                    "confidence": 0.9,
                    "updatedAt": NOW.isoformat(),
                    "insightKind": "value",
                    "aspect": "criterion",
                }
            ],
            "dislikes": [{"value": "남에게 맞추는 삶", "insightKind": "life_attitude"}],
        }
    }
    signals, guide = recommendation_inputs(payload, now=NOW)
    assert signals == ()
    assert guide.preferences == ()
    assert guide.exclusions == ()


def test_negative_personal_principle_keeps_its_reason():
    utterance = "남의 기대에만 맞추는 건 싫어요. 제 선택이 없어지는 것 같아서요."
    merged = merge_item(
        value_payload(
            field="dislikes",
            value="남의 기대에만 맞추기",
            evidence="남의 기대에만 맞추는 건 싫어요",
            context=None,
            origin=None,
            reason="제 선택이 없어지는 것 같아서요",
        ),
        utterance,
    )
    (signal,) = merged.accepted
    assert signal.field == TasteField.DISLIKES
    assert signal.insight_kind == InsightKind.VALUE
    assert signal.reason.text == "제 선택이 없어지는 것 같아서요"
