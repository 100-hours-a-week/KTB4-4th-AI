"""추천 이유 문구.

이미 계산된 값만 조합한다. 어떤 관심사로 찾았는지, 여러 관심사에 함께 걸렸는지, 취향 가산을
받았는지, 탐색 추천인지를 보고 문형을 고른다. 상품의 특징은 말하지 않는다. 상품 문서가 상품명만
보고 쓴 것이라 확인되지 않은 장점을 붙이게 되기 때문이다.

선물 목록은 친구가 본다. 관심사 이름은 친구 이상 공개인 관심사·취미와 취향만 쓰고, 갖고 싶지만
못 산 것은 무엇인지 말하지 않는다. 받는 사람 이름은 username 원문에 "님"을 붙인다.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.profile.models import TasteField, Visibility
from app.domain.recommendation.models import RecommendationMode, ScoredMatch

_SHOWN_TO_FRIENDS = frozenset({Visibility.FRIENDS, Visibility.PUBLIC})
_NAMED_INTEREST_FIELDS = frozenset({TasteField.INTERESTS, TasteField.HOBBIES})
# 두 관심사를 함께 짚는 이유는 두 매치가 모두 이만큼 가까울 때만 쓴다. 상위 80개 안에 약하게
# 걸린 것까지 세면 디지털카메라에 "차와 콘솔게임을 함께 좋아하는" 같은 이유가 붙는다.
SHARED_REASON_MIN_SIMILARITY = 0.6
# 숫자를 한국어로 읽을 때 받침이 있는 것: 0(영), 1(일), 3(삼), 6(육), 7(칠), 8(팔)
_DIGITS_WITH_FINAL = frozenset("013678")

# 같은 목록에서 이웃한 상품의 이유가 같은 문장이 되지 않게 순위로 문형을 돌린다.
# 첫 문형이 1등 상품에 쓰인다.
_SELF_INTEREST = (
    "{v}에 대한 관심과 잘 맞는 상품이에요.",
    "{v_eul} 좋아하는 취향에 맞는 상품이에요.",
)
_SELF_HOBBY = (
    "{v}에 대한 관심과 잘 맞는 상품이에요.",
    "{v_eul} 즐길 때 챙기기 좋은 상품이에요.",
)
_GIFT_INTEREST = (
    "{v}에 관심 있는 {who}에게 선물하기 좋은 상품이에요.",
    "{v_eul} 즐기는 {who}에게 잘 어울리는 선물이에요.",
)


def _has_final(word: str) -> bool:
    last = word.strip()[-1:] if word.strip() else ""
    if "가" <= last <= "힣":
        return (ord(last) - ord("가")) % 28 != 0
    return last in _DIGITS_WITH_FINAL


def _eul(word: str) -> str:
    return f"{word}{'을' if _has_final(word) else '를'}"


def _wa(word: str) -> str:
    return f"{word}{'과' if _has_final(word) else '와'}"


def _shown(match: ScoredMatch, mode: RecommendationMode) -> bool:
    """이유 문구에 관심사 이름을 넣어도 되는지. 본인 목록은 늘 된다."""
    signal = match.match.query.signal
    if mode == RecommendationMode.SELF or signal.field == TasteField.PREFERENCES:
        return True
    return signal.field in _NAMED_INTEREST_FIELDS and signal.visibility in _SHOWN_TO_FRIENDS


def _shared_interests(matches: Sequence[ScoredMatch], mode: RecommendationMode) -> list[str]:
    """이 상품이 함께 걸린 관심사·취미 중 이름을 써도 되는 것. 기여도 순서다."""
    values: list[str] = []
    for scored in matches:
        signal = scored.match.query.signal
        if scored.match.query.exploration or signal.field not in _NAMED_INTEREST_FIELDS:
            continue
        if scored.match.similarity < SHARED_REASON_MIN_SIMILARITY:
            continue
        if _shown(scored, mode) and signal.value not in values:
            values.append(signal.value)
    return values


def recommendation_reason(
    matches: Sequence[ScoredMatch],
    mode: RecommendationMode,
    *,
    rank: int = 0,
    recipient_name: str | None = None,
    boost_label: str | None = None,
) -> str:
    """matches 는 기여도 순으로 정렬된 이 상품의 매치다.

    관심사 쿼리로 찾은 매치가 있으면 그중 첫 매치가 이유의 기준이다. 탐색 쿼리로만 찾은 상품만
    탐색 추천 이유를 쓴다. 관심사 쿼리에도 걸린 상품은 관심사 상품으로 목록에 들어가기 때문이다.
    """
    best = next((scored for scored in matches if not scored.match.query.exploration), matches[0])
    signal = best.match.query.signal
    value = signal.value
    names = {"v": value, "v_eul": _eul(value)}
    who = f"{recipient_name}님" if recipient_name else "분"
    owner = f"{recipient_name}님의 " if recipient_name else ""
    gift = mode == RecommendationMode.GIFT

    def pick(variants: Sequence[str]) -> str:
        return variants[rank % len(variants)].format(who=who, **names)

    if best.match.query.exploration:
        if not gift:
            return f"{_eul(value)} 좋아하는 취향을 반영한 새로운 추천이에요."
        if _shown(best, mode):
            return f"{_eul(value)} 좋아하는 {who}에게 새롭게 제안하는 상품이에요."
        return f"{owner}취향과 관심사를 반영한 새로운 추천이에요."

    shared = _shared_interests(matches, mode)
    if len(shared) >= 2:
        first, second = shared[:2]
        if gift:
            reason = f"{_wa(first)} {_eul(second)} 함께 좋아하는 {who}에게 잘 어울리는 선물이에요."
        else:
            reason = f"{_wa(first)} {_eul(second)} 함께 즐기는 취향에 맞는 상품이에요."
    elif signal.field == TasteField.PREFERENCES:
        # 취향 쿼리로 찾은 상품. 취향은 친구 공개 항목이라 선물 목록에서도 그대로 보여 준다.
        if gift:
            return f"{owner}{value} 취향에 맞춰 선물하기 좋은 상품이에요."
        return f"{value} 취향과 잘 맞는 상품이에요."
    elif gift:
        if signal.field == TasteField.UNAFFORDABLE:
            # 무엇을 못 샀는지는 친구에게 보이지 않는다.
            reason = "스스로 사기엔 망설여지는 물건이라 선물로 반가울 거예요."
        elif _shown(best, mode):
            reason = pick(_GIFT_INTEREST)
        else:
            reason = f"{owner}평소 관심사와 잘 맞아 선물하기 좋은 상품이에요."
    elif signal.field == TasteField.HOBBIES:
        reason = pick(_SELF_HOBBY)
    elif signal.field == TasteField.WANTS:
        reason = f"갖고 싶다고 한 {_wa(value)} 가까운 상품이에요."
    elif signal.field == TasteField.CONSUMABLES:
        reason = f"자주 쓰는 {_eul(value)} 채워 두기 좋은 상품이에요."
    else:
        reason = pick(_SELF_INTEREST)

    if boost_label and boost_label != value:
        reason = f"{reason} {boost_label} 취향도 반영했어요."
    return reason
