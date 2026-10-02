"""판단 모델(TypeSafe Jev)로 취향을 추출한다.

추출 모델은 발화에 나온 대상(후보)만 찾고, 후보마다 사용자의 태도는 판단 모델이 확률로 답한다.
어느 항목(field)에 넣을지는 태도와 대상 종류를 보고 코드가 정한다. 추출 모델이 칸부터 고르면
"러닝은 싫어요"의 러닝이 관심사로 들어가는 식의 오류를 막을 방법이 없어서 역할을 나눴다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from app.domain.conversation.models import ConversationGoal, ConversationState, ConversationStyle
from app.domain.profile.merger import normalize_text
from app.domain.profile.models import (
    DeferralReason,
    DropRef,
    EvidenceType,
    ExtractedItem,
    ExtractionDelta,
    PreferenceAspect,
    ProfileSignal,
    ProfileState,
    TasteField,
)
from app.domain.profile.taxonomy import (
    INTEREST_CATEGORIES,
    INTEREST_CATEGORIES_BY_KEY,
    TASTE_AXES,
    TASTE_AXES_BY_KEY,
)

# 추출 모델은 놓치지 않는 쪽으로 넉넉히 뽑고, 판단 모델이 거른다.
MAX_CANDIDATES = 8
MAX_EVIDENCE_LENGTH = 40
# 판단 모델의 confidence가 이보다 낮은 태도는 저장하지 않는다. 실제 대화 로그를 보고 조정한다.
MIN_STANCE_CONFIDENCE = 0.6
# 취향은 "보통 혼자 봐요"처럼 습관으로 드러나 태도 확신이 낮게 나오기 쉽다.
# 취향 축까지 잡힌 후보는 기준을 낮춰 받는다.
MIN_TASTE_STANCE_CONFIDENCE = 0.5
MIN_DEFERRAL_CONFIDENCE = 0.6
HABITUAL_THRESHOLD = 0.5
EXPLICIT_THRESHOLD = 0.5
NONE_ANSWER_THRESHOLD = 0.7
# 취향·관심사로 볼 만한지(material)와 분류(축, 분야) 판단을 받아들이는 confidence.
MIN_MATERIAL_CONFIDENCE = 0.6
MIN_TAXONOMY_CONFIDENCE = 0.5
# 구체성 점수(0~2)가 이보다 낮은 관심사는 "운동"처럼 대분류 수준이라 신뢰도를 낮춰 순위를 뒤로 뺀다.
BROAD_SPECIFICITY = 0.7
BROAD_INTEREST_CONFIDENCE_CAP = 0.55
# 하고 싶다는 활동은 희망일 뿐이라 관심사로 넣되 신뢰도를 낮춘다.
WISH_CONFIDENCE_CAP = 0.5
# 한 글자 값은 다른 단어 안에 우연히 들어가는 경우가 많아 재언급 후보로 보지 않는다.
MIN_REMENTION_LENGTH = 2
PREVIOUS_USER_MESSAGES = 2
# 정정 판단에 올리는 프로필 항목 수.
MAX_CORRECTION_TARGETS = 6
# 직전 이 턴 수 안에 들어온 항목은 "아까 그거"처럼 대상을 말하지 않아도 정정 대상 후보가 된다.
RECENT_CORRECTION_TURNS = 2
# 정정은 턴 단위 의도와 항목별 취소 판단이 둘 다 넘어야 반영한다.
# 항목 판단만 쓰면 항목을 다시 언급하기만 해도 취소로 읽힐 수 있다.
CORRECTION_INTENT_THRESHOLD = 0.6
RETRACT_THRESHOLD = 0.7
# 싫은 것·제약은 추천에서 상품을 빼는 안전 정보라 더 확실할 때만 내린다.
SAFETY_RETRACT_THRESHOLD = 0.85
_ANSWER_DEPTH_CRITERIA = [
    "Brushes the question off or answers minimally",
    "Answers with plain facts",
    "Shares reasons, feelings, or what matters to them",
]


class CandidateKind(StrEnum):
    ACTIVITY = "activity"
    THING = "thing"
    ATTRIBUTE = "attribute"
    CONTEXT = "context"


class Stance(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"
    WANT = "want"
    OWNED = "owned"
    REPURCHASE = "repurchase"
    CANNOT_USE = "cannot_use"
    LIFE_CONTEXT = "life_context"
    USED_TO_LIKE = "used_to_like"
    MENTION_ONLY = "mention_only"
    OTHER_PERSON = "other_person"
    NOT_A_SUBJECT = "not_a_subject"


# Jev는 영어로 주로 학습했고 질문을 글자 그대로 읽는다. 선택지마다 조건을 영어로 분명히 적는다.
_STANCE_CRITERIA = {
    Stance.LIKE: ("The user personally likes, enjoys, is into, or regularly chooses to do it now"),
    Stance.DISLIKE: (
        "The user personally dislikes it, finds it unpleasant or bothersome, or wants to avoid it"
    ),
    Stance.WANT: "The user wants to have it, buy it, or try it in the future",
    Stance.OWNED: "The user already owns or has it",
    Stance.REPURCHASE: "It is a consumable the user uses up and buys again",
    Stance.CANNOT_USE: (
        "The user cannot use or eat it because of health, allergy, size, or a similar limit"
    ),
    Stance.LIFE_CONTEXT: (
        "It describes the user's living situation or routine, such as living alone or working "
        "from home, without a like or dislike"
    ),
    Stance.USED_TO_LIKE: "The user liked or did it in the past but no longer does",
    Stance.MENTION_ONLY: "The user mentions it without a clear attitude toward it",
    Stance.OTHER_PERSON: "It is about someone else's taste or activity, not the user's own",
    Stance.NOT_A_SUBJECT: (
        "It is a reason, feeling, or bodily reaction the user mentioned, such as being out of "
        "breath or finding something tedious, not a thing or activity"
    ),
}


class Material(StrEnum):
    INTEREST = "interest"
    TASTE = "taste"
    LIFE_CONTEXT = "life_context"
    NOT_MATERIAL = "not_material"


# 좋고 싫음과 무관하게 묻는다. "취향"이라는 말을 넣으면 싫다는 발화와 충돌해 판단이 흔들린다.
_MATERIAL_CRITERIA = {
    Material.INTEREST: (
        "A specific field, thing, or activity the user personally does, uses, likes, or avoids "
        "over time"
    ),
    Material.TASTE: (
        "A quality, style, sensory feature, setting, or way of doing things that the user "
        "prefers or avoids, such as a color, a flavor, doing things alone, or quiet places"
    ),
    Material.LIFE_CONTEXT: "The user's ongoing living situation or routine that affects what "
    "they need",
    Material.NOT_MATERIAL: (
        "Too temporary, incidental, or not about the user to reveal what they like, such as "
        "today's meal, the weather, being busy, a feeling, a reason, or someone else's taste"
    ),
}
_TASTE_AXIS_CRITERIA = {
    **{axis.key: f"{axis.group}/{axis.name}: {axis.description}" for axis in TASTE_AXES},
    "none": "None of these; it is not a preference about quality, style, or way of doing things",
}
_INTEREST_CATEGORY_CRITERIA = {
    **{category.key: category.description for category in INTEREST_CATEGORIES},
    "none": "None of these fields",
}
_SPECIFICITY_CRITERIA = [
    "Only a broad field, such as exercise, music, or food",
    "Somewhat specific, such as a kind of activity or genre",
    "Specific enough to search for concrete gifts, such as a particular sport, genre, or item",
]
_SAFETY_FIELDS = frozenset({TasteField.DISLIKES, TasteField.CONSTRAINTS})
_POSITIVE_FIELDS = frozenset(
    {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.PREFERENCES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    }
)
# 정정 판단 때 프로필 항목이 어떤 정보인지 판단 모델에 알려 주는 설명.
_FIELD_DESCRIPTIONS = {
    TasteField.INTERESTS: "something the user is interested in",
    TasteField.HOBBIES: "an activity the user does regularly",
    TasteField.PREFERENCES: "a style or way of doing things the user prefers",
    TasteField.LIFESTYLE: "the user's living situation",
    TasteField.WANTS: "something the user wants to have",
    TasteField.UNAFFORDABLE: "something the user wants but has not bought",
    TasteField.CONSUMABLES: "something the user uses up and buys again",
    TasteField.OWNED: "something the user already owns",
    TasteField.DISLIKES: "something the user dislikes",
    TasteField.CONSTRAINTS: "something the user cannot use",
}
_QUERY_FIELDS = frozenset(
    {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    }
)

# 취향 후보는 대상이 아니라 선호라서 "가졌다", "산다" 같은 선택지가 판단을 흐린다.
# 같은 키를 쓰되 선호의 방향만 묻는다.
_TASTE_STANCE_CRITERIA = {
    Stance.LIKE: (
        "The user prefers, enjoys, deliberately chooses, or usually does things this way"
    ),
    Stance.DISLIKE: "The user dislikes or avoids this quality, style, or way of doing things",
    Stance.USED_TO_LIKE: "The user preferred it in the past but no longer does",
    Stance.MENTION_ONLY: "The user mentions it without showing a preference for or against it",
    Stance.OTHER_PERSON: "It is someone else's preference, not the user's own",
}

_DEFERRAL_CRITERIA = {
    "none": "Nothing is holding the user back from getting it",
    DeferralReason.PRICE: "It is too expensive for the user",
    DeferralReason.JUSTIFICATION: (
        "The user feels they cannot justify buying it for themselves or that it is unnecessary"
    ),
    DeferralReason.TIMING: "The user is waiting for the right time or occasion",
}
_KIND_BY_FIELD = {
    TasteField.HOBBIES: CandidateKind.ACTIVITY,
    TasteField.PREFERENCES: CandidateKind.ATTRIBUTE,
    TasteField.LIFESTYLE: CandidateKind.CONTEXT,
}


@dataclass(slots=True, frozen=True)
class Candidate:
    subject: str
    kind: CandidateKind
    aspect: PreferenceAspect | None = None
    target: str | None = None


@dataclass(slots=True, frozen=True)
class Judgment:
    """후보 하나에 대한 판단 결과. 로그와 디버깅용이다."""

    candidate: Candidate
    stance: Stance
    confidence: float
    field: TasteField | None
    material: Material | None = None
    taxonomy_path: tuple[str, ...] | None = None


@dataclass(slots=True, frozen=True)
class JudgmentRequest:
    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]
    sentences: tuple[str, ...]
    # 정정 판단에 올린 프로필 항목. retract_{k}의 k가 이 순서다.
    targets: tuple[ProfileSignal, ...] = ()


ShortValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20),
]
AxisValue = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=40),
]


class _CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidatePayload(_CandidateModel):
    subject: ShortValue
    kind: CandidateKind
    aspect: PreferenceAspect | None = None
    target: ShortValue | None = None


class CandidateDeltaPayload(_CandidateModel):
    """추출 모델에 안내하는 출력 스키마. 실제 파싱은 parse_candidates가 항목 단위로 한다."""

    candidates: list[CandidatePayload] = Field(default_factory=list, max_length=MAX_CANDIDATES)
    axes: list[AxisValue] = Field(default_factory=list, max_length=3)


def parse_candidates(raw: object) -> tuple[list[Candidate], list[str], list[str]]:
    """후보를 항목 단위로 검증한다. 최상위가 객체가 아닐 때만 예외를 던져 재시도하게 한다."""
    if not isinstance(raw, Mapping):
        raise ValueError("candidate result is not an object")

    dropped: list[str] = []
    candidates: list[Candidate] = []
    seen: set[str] = set()
    raw_candidates = raw.get("candidates")
    for raw_item in raw_candidates if isinstance(raw_candidates, list) else []:
        if len(candidates) >= MAX_CANDIDATES:
            dropped.append("candidate limit exceeded")
            break
        if not isinstance(raw_item, Mapping):
            dropped.append("candidate is not an object")
            continue
        item = dict(raw_item)
        if isinstance(item.get("target"), str) and not item["target"].strip():
            item["target"] = None
        try:
            payload = CandidatePayload.model_validate(item)
        except ValidationError as error:
            dropped.append(f"{raw_item.get('subject')!r}: {error.errors()[0]['msg']}")
            continue
        normalized = normalize_text(payload.subject)
        if normalized in seen:
            continue
        seen.add(normalized)
        is_attribute = payload.kind == CandidateKind.ATTRIBUTE
        candidates.append(
            Candidate(
                subject=payload.subject,
                kind=payload.kind,
                aspect=payload.aspect if is_attribute else None,
                target=payload.target if is_attribute else None,
            )
        )

    axes: list[str] = []
    raw_axes = raw.get("axes")
    for axis in raw_axes if isinstance(raw_axes, list) else []:
        if isinstance(axis, str) and 0 < len(axis.strip()) <= MAX_EVIDENCE_LENGTH:
            axes.append(axis.strip())
    return candidates, axes[:3], dropped


def remention_candidates(
    profile: ProfileState,
    utterance: str,
    known: Sequence[Candidate],
) -> list[Candidate]:
    """이미 프로필에 있는 대상이 이번 발화에 다시 나오면 추출 모델이 놓쳐도 판단 대상에 넣는다."""
    text = normalize_text(utterance)
    seen = {normalize_text(candidate.subject) for candidate in known}
    found: list[Candidate] = []
    for signal in profile.active_signals():
        value = signal.normalized_value
        if len(value) < MIN_REMENTION_LENGTH or value in seen or value not in text:
            continue
        seen.add(value)
        found.append(
            Candidate(
                subject=signal.value,
                kind=_KIND_BY_FIELD.get(signal.field, CandidateKind.THING),
                aspect=signal.aspect,
                target=signal.target,
            )
        )
    return found


def correction_targets(state: ConversationState, utterance: str) -> list[ProfileSignal]:
    """이번 발화가 고칠 수 있는 프로필 항목을 고른다.

    값이나 붙은 관심사가 이번 발화·직전 발화에 나온 항목을 먼저 두고,
    "아까 그거"처럼 대상을 말하지 않는 경우를 위해 최근에 들어온 항목을 뒤에 둔다.
    """
    previous_users = [turn.content for turn in state.history if turn.role == "user"][
        -PREVIOUS_USER_MESSAGES:
    ]
    text = normalize_text(" ".join([*previous_users, utterance]))
    stored = state.profile.stored_signals()
    # 직전 AI 턴이 되비추기였으면 그때 짚은 항목이 "아니에요"의 대상이다.
    reflected: list[ProfileSignal] = []
    if state.last_goal == ConversationGoal.REFLECT:
        values = {normalize_text(value) for value in state.reflection_values}
        reflected = [signal for signal in stored if signal.normalized_value in values]

    def mentioned_in_text(signal: ProfileSignal) -> bool:
        value = signal.normalized_value
        if len(value) >= MIN_REMENTION_LENGTH and value in text:
            return True
        return bool(signal.target) and normalize_text(signal.target or "") in text

    mentioned = [signal for signal in stored if mentioned_in_text(signal)]
    recent_turn = state.turn_count - RECENT_CORRECTION_TURNS + 1
    recent = sorted(
        (signal for signal in stored if signal.source_turn >= recent_turn),
        key=lambda signal: signal.source_turn,
        reverse=True,
    )
    targets: list[ProfileSignal] = []
    for signal in [*reflected, *mentioned, *recent]:
        if signal not in targets:
            targets.append(signal)
    return targets[:MAX_CORRECTION_TARGETS]


# 관심사 앞에 빈도나 때만 덧붙인 말. "주말 캠핑"은 "캠핑"과 같은 관심사다.
_FREQUENCY_WORDS = frozenset(
    {"주말", "주말마다", "평일", "매일", "매주", "요즘", "가끔", "자주", "종종", "퇴근", "퇴근 후"}
)


def without_frequency_variants(
    candidates: Sequence[Candidate],
    *,
    known: set[str] | frozenset[str] = frozenset(),
) -> list[Candidate]:
    """같은 관심사에 빈도나 때만 붙인 후보("주말 캠핑")를 뺀다.

    추출 모델이 프롬프트 규칙을 놓칠 때의 안전망이다.
    """
    # 이번 후보뿐 아니라 이미 저장된 관심사("핸드드립 커피")와도 비교한다.
    subjects = {normalize_text(candidate.subject) for candidate in candidates} | set(known)
    kept: list[Candidate] = []
    for candidate in candidates:
        words = normalize_text(candidate.subject).split()
        variant = any(
            len(words) > cut
            and " ".join(words[:cut]) in _FREQUENCY_WORDS
            and " ".join(words[cut:]) in subjects
            for cut in (1, 2)
        )
        if not variant:
            kept.append(candidate)
    return kept


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…~])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def build_judgment_request(
    state: ConversationState,
    utterance: str,
    candidates: Sequence[Candidate],
    targets: Sequence[ProfileSignal] = (),
) -> JudgmentRequest:
    # 판단은 이번 턴을 history에 넣기 전에 실행되므로 마지막 assistant 메시지가 직전 질문이다.
    previous_ai = next(
        (turn.content for turn in reversed(state.history) if turn.role == "assistant"),
        "",
    )
    previous_users = [turn.content for turn in state.history if turn.role == "user"][
        -PREVIOUS_USER_MESSAGES:
    ]
    # 근거 문장은 이번 발화를 먼저 두고, 앞 발화를 받아서 한 말일 수 있어 직전 발화도 넣는다.
    sentences = split_sentences(utterance) or [utterance.strip()]
    if previous_users:
        sentences.extend(split_sentences(previous_users[-1]))

    judgment_state: dict[str, Any] = {
        "previous_ai_message": previous_ai,
        "previous_user_messages": previous_users,
        "user_message": utterance,
    }
    questions: dict[str, dict[str, Any]] = {}
    if previous_ai:
        questions["none_answer"] = {
            "type": "noul",
            "instructions": (
                "Does `user_message` clearly answer that nothing applies to the question in "
                "`previous_ai_message`, such as 'nothing in particular' or 'there wasn't any'?"
            ),
        }
    for index, candidate in enumerate(candidates):
        subject = candidate.subject
        questions[f"stance_{index}"] = {
            "type": "choice",
            "instructions": {
                "candidate": subject,
                "question": (
                    "What is the user's own attitude toward `candidate`? Judge only by what the "
                    "user said in `user_message`, or in `previous_user_messages` when "
                    "`user_message` refers back to them. `previous_ai_message` only helps to "
                    "understand short answers; its wording is not the user's attitude."
                ),
            },
            "criteria": {
                stance.value: text
                for stance, text in (
                    _TASTE_STANCE_CRITERIA
                    if candidate.kind == CandidateKind.ATTRIBUTE
                    else _STANCE_CRITERIA
                ).items()
            },
        }
        questions[f"material_{index}"] = {
            "type": "choice",
            "instructions": {
                "candidate": subject,
                "question": (
                    "Regardless of whether the user likes or dislikes it, what kind of clue is "
                    "`candidate` in `user_message` about the user's own life?"
                ),
            },
            "criteria": {material.value: text for material, text in _MATERIAL_CRITERIA.items()},
        }
        if candidate.kind == CandidateKind.ATTRIBUTE:
            questions[f"taste_axis_{index}"] = {
                "type": "choice",
                "instructions": {
                    "candidate": subject,
                    "question": "Which dimension of taste does `candidate` describe?",
                },
                "criteria": _TASTE_AXIS_CRITERIA,
            }
        if candidate.kind in {CandidateKind.ACTIVITY, CandidateKind.THING}:
            questions[f"interest_category_{index}"] = {
                "type": "choice",
                "instructions": {
                    "candidate": subject,
                    "question": "Which field of interest does `candidate` belong to?",
                },
                "criteria": _INTEREST_CATEGORY_CRITERIA,
            }
            questions[f"specificity_{index}"] = {
                "type": "score",
                "instructions": {
                    "candidate": subject,
                    "question": "How specific is `candidate` as a clue for choosing a gift?",
                },
                "criteria": _SPECIFICITY_CRITERIA,
            }
        questions[f"explicit_{index}"] = {
            "type": "noul",
            "instructions": {
                "candidate": subject,
                "question": (
                    "Does the user say what they do or feel about `candidate` directly, "
                    "rather than it having to be inferred?"
                ),
            },
        }
        if candidate.kind == CandidateKind.ACTIVITY:
            questions[f"habitual_{index}"] = {
                "type": "noul",
                "instructions": {
                    "candidate": subject,
                    "question": "Does the user say they personally do `candidate` regularly?",
                },
            }
        if candidate.kind == CandidateKind.THING:
            questions[f"deferral_{index}"] = {
                "type": "choice",
                "instructions": {
                    "candidate": subject,
                    "question": (
                        "If the user wants `candidate` but has not gotten it, what is holding "
                        "them back?"
                    ),
                },
                "criteria": {str(key): text for key, text in _DEFERRAL_CRITERIA.items()},
            }
        if len(sentences) > 1:
            questions[f"evidence_{index}"] = {
                "type": "choice",
                "instructions": {
                    "candidate": subject,
                    "question": "Which sentence best shows the user's attitude toward `candidate`?",
                },
                "criteria": {f"s{position}": text for position, text in enumerate(sentences)},
            }
    if previous_ai and state.conversation_style in {
        ConversationStyle.REFLECTIVE,
        ConversationStyle.COMPANION,
    }:
        # 대화 속도 조절용. reflective는 짧게 넘기는 답이면 가볍게 돌아오고,
        # companion은 사용자 에너지 추정에 더한다.
        questions["answer_depth"] = {
            "type": "score",
            "instructions": (
                "How much does `user_message` share in reply to `previous_ai_message`?"
            ),
            "criteria": _ANSWER_DEPTH_CRITERIA,
        }
    if targets:
        judgment_state["profile_items"] = {
            f"p{index}": f"{signal.value} ({_FIELD_DESCRIPTIONS[signal.field]})"
            for index, signal in enumerate(targets)
        }
        questions["correction_intent"] = {
            "type": "noul",
            "instructions": (
                "Does `user_message` take back or correct something the user said earlier, "
                "or something the AI assumed about the user?"
            ),
        }
        for index, signal in enumerate(targets):
            questions[f"retract_{index}"] = {
                "type": "noul",
                "instructions": {
                    "item": f"{signal.value} ({_FIELD_DESCRIPTIONS[signal.field]})",
                    "question": (
                        "Does `user_message` say that `item` is wrong, was only a passing "
                        "remark, or is no longer true for the user?"
                    ),
                },
            }
    return JudgmentRequest(
        state=judgment_state,
        questions=questions,
        sentences=tuple(sentences),
        targets=tuple(targets),
    )


def judgment_delta(
    candidates: Sequence[Candidate],
    answers: Mapping[str, Any],
    sentences: Sequence[str],
    *,
    axes: Sequence[str] = (),
    targets: Sequence[ProfileSignal] = (),
) -> tuple[ExtractionDelta, list[Judgment]]:
    items: list[ExtractedItem] = []
    judgments: list[Judgment] = []
    drops = _retractions(answers, targets)
    for index, candidate in enumerate(candidates):
        answer = answers.get(f"stance_{index}")
        if not isinstance(answer, Mapping):
            continue
        try:
            stance = Stance(str(answer.get("choice")))
        except ValueError:
            continue
        confidence = _number(answer.get("confidence"))
        material = _material(answers.get(f"material_{index}"))
        path, axis_aspect = _taxonomy(answers, index)
        field: TasteField | None = None
        deferral: DeferralReason | None = None
        minimum = (
            MIN_TASTE_STANCE_CONFIDENCE
            if candidate.kind == CandidateKind.ATTRIBUTE and axis_aspect is not None
            else MIN_STANCE_CONFIDENCE
        )
        if confidence >= minimum:
            field, deferral = _field_for(
                candidate,
                stance,
                material,
                answers,
                index,
                has_taste_axis=axis_aspect is not None,
            )
        judgments.append(Judgment(candidate, stance, confidence, field, material, path))
        if stance == Stance.USED_TO_LIKE and confidence >= MIN_STANCE_CONFIDENCE:
            # "뜨개질 요즘은 안 해요": 예전 취향으로 남아 있던 항목을 내린다.
            normalized = normalize_text(candidate.subject)
            drops.extend(
                DropRef(field=signal.field, value=signal.value)
                for signal in targets
                if signal.field in _POSITIVE_FIELDS and signal.normalized_value == normalized
            )
        if field is None:
            continue

        probabilities = answer.get("probabilities")
        item_confidence = confidence
        if isinstance(probabilities, Mapping):
            item_confidence = _number(probabilities.get(stance.value), default=confidence)
        explicit = _noul(answers, f"explicit_{index}") >= EXPLICIT_THRESHOLD
        evidence_type = EvidenceType.EXPLICIT if explicit else EvidenceType.INFERRED
        if stance == Stance.WANT and candidate.kind == CandidateKind.ACTIVITY:
            evidence_type = EvidenceType.INFERRED
            item_confidence = min(item_confidence, WISH_CONFIDENCE_CAP)
        specificity = _score(answers.get(f"specificity_{index}"))
        if field in _QUERY_FIELDS and specificity is not None and specificity < BROAD_SPECIFICITY:
            item_confidence = min(item_confidence, BROAD_INTEREST_CONFIDENCE_CAP)
        is_preference = field == TasteField.PREFERENCES
        items.append(
            ExtractedItem(
                field=field,
                value=candidate.subject,
                confidence=item_confidence,
                evidence=_evidence(answers.get(f"evidence_{index}"), sentences),
                evidence_type=evidence_type,
                deferral_reason=deferral,
                aspect=(axis_aspect or candidate.aspect) if is_preference else None,
                target=candidate.target if is_preference else None,
                taxonomy_path=path,
            )
        )

    delta = ExtractionDelta(
        items=tuple(items),
        axes=tuple(axes),
        drop=tuple(dict.fromkeys(drops)),
        none_answer=_noul(answers, "none_answer") >= NONE_ANSWER_THRESHOLD,
        answer_depth=_score(answers.get("answer_depth")),
    )
    return delta, judgments


def _retractions(
    answers: Mapping[str, Any],
    targets: Sequence[ProfileSignal],
) -> list[DropRef]:
    if not targets or _noul(answers, "correction_intent") < CORRECTION_INTENT_THRESHOLD:
        return []
    return [
        DropRef(field=signal.field, value=signal.value)
        for index, signal in enumerate(targets)
        if _noul(answers, f"retract_{index}")
        >= (SAFETY_RETRACT_THRESHOLD if signal.field in _SAFETY_FIELDS else RETRACT_THRESHOLD)
    ]


def _field_for(
    candidate: Candidate,
    stance: Stance,
    material: Material | None,
    answers: Mapping[str, Any],
    index: int,
    *,
    has_taste_axis: bool = False,
) -> tuple[TasteField | None, DeferralReason | None]:
    # 못 쓰는 것, 가진 것, 다시 사는 것은 추천 제외와 중복 방지에 필요해서 분류와 상관없이 남긴다.
    if stance == Stance.CANNOT_USE:
        return TasteField.CONSTRAINTS, None
    if stance == Stance.OWNED:
        return TasteField.OWNED, None
    if stance == Stance.REPURCHASE:
        return TasteField.CONSUMABLES, None
    # 오늘 먹은 것, 날씨, 다른 사람 이야기처럼 취향·관심사의 단서가 아니면
    # 좋다 싫다를 말해도 저장하지 않는다.
    if material == Material.NOT_MATERIAL:
        return None, None
    if stance == Stance.DISLIKE:
        return TasteField.DISLIKES, None
    if stance == Stance.LIFE_CONTEXT:
        return TasteField.LIFESTYLE, None

    kind = candidate.kind
    # 취향 후보가 취향 축으로 분류됐으면 취향이다. "혼자 가는 캠핑"처럼 관심사 이름이 들어간
    # 취향을 material이 관심사로 읽는 경우가 있어서 축 판단을 우선한다.
    # 둘 다 확신하지 못하면 추출 모델이 붙인 종류를 따른다.
    is_attribute = kind == CandidateKind.ATTRIBUTE
    is_taste = (
        material == Material.TASTE
        or (is_attribute and has_taste_axis)
        or (material is None and is_attribute)
    )
    is_context = material == Material.LIFE_CONTEXT or (
        material is None and kind == CandidateKind.CONTEXT
    )
    if stance == Stance.WANT:
        if is_taste:
            return TasteField.PREFERENCES, None
        if is_context:
            return None, None
        if kind == CandidateKind.ACTIVITY:
            return TasteField.INTERESTS, None
        deferral = _deferral(answers.get(f"deferral_{index}"))
        if deferral is not None:
            return TasteField.UNAFFORDABLE, deferral
        return TasteField.WANTS, None
    if stance == Stance.LIKE:
        if is_taste:
            return TasteField.PREFERENCES, None
        if is_context:
            return TasteField.LIFESTYLE, None
        if (
            kind == CandidateKind.ACTIVITY
            and _noul(answers, f"habitual_{index}") >= HABITUAL_THRESHOLD
        ):
            return TasteField.HOBBIES, None
        return TasteField.INTERESTS, None
    # 예전에 좋아했던 것, 언급만 한 것, 다른 사람 이야기, 이유나 감정은 저장하지 않는다.
    return None, None


def _material(answer: object) -> Material | None:
    choice = _confident_choice(answer, MIN_MATERIAL_CONFIDENCE)
    try:
        return Material(choice) if choice else None
    except ValueError:
        return None


def _taxonomy(
    answers: Mapping[str, Any],
    index: int,
) -> tuple[tuple[str, ...] | None, PreferenceAspect | None]:
    axis = TASTE_AXES_BY_KEY.get(
        _confident_choice(answers.get(f"taste_axis_{index}"), MIN_TAXONOMY_CONFIDENCE) or ""
    )
    if axis is not None:
        return axis.path, axis.aspect
    category = INTEREST_CATEGORIES_BY_KEY.get(
        _confident_choice(answers.get(f"interest_category_{index}"), MIN_TAXONOMY_CONFIDENCE) or ""
    )
    if category is not None:
        return category.path, None
    return None, None


def _confident_choice(answer: object, minimum: float) -> str | None:
    if not isinstance(answer, Mapping) or _number(answer.get("confidence")) < minimum:
        return None
    choice = answer.get("choice")
    return str(choice) if choice is not None else None


def _score(answer: object) -> float | None:
    if not isinstance(answer, Mapping):
        return None
    value = answer.get("score")
    return None if isinstance(value, bool) or not isinstance(value, int | float) else float(value)


def _deferral(answer: object) -> DeferralReason | None:
    if not isinstance(answer, Mapping):
        return None
    if _number(answer.get("confidence")) < MIN_DEFERRAL_CONFIDENCE:
        return None
    try:
        return DeferralReason(str(answer.get("choice")))
    except ValueError:
        return None


def _evidence(answer: object, sentences: Sequence[str]) -> str:
    chosen = sentences[0]
    if isinstance(answer, Mapping):
        match = re.fullmatch(r"s(\d+)", str(answer.get("choice")))
        if match is not None and int(match.group(1)) < len(sentences):
            chosen = sentences[int(match.group(1))]
    return chosen[:MAX_EVIDENCE_LENGTH].strip()


def _noul(answers: Mapping[str, Any], key: str) -> float:
    answer = answers.get(key)
    return _number(answer.get("noul")) if isinstance(answer, Mapping) else 0.0


def _number(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)
