"""분석 화면의 취향·관심사 키워드.

저장된 프로필 값은 추천 검색에 맞춰 "포근한 향 바디로션"처럼 대상까지 붙인 구라서 그대로 보여 주면
길고, 같은 결의 값이 겹친다. 키워드는 모델이 저장 값들을 묶어 다시 이름 붙인 것을 쓰고,
여기서는 그 결과가 근거 항목과 맞는지 검증한다.
모델을 쓰지 못하면 저장 값에서 대상 이름만 떼어 쓴다.

- 취향: 무엇을 좋아하는가. 대상이 아니라 좋아하는 결(향, 맛, 분위기, 방식)이다.
- 관심사: 무엇에 관심을 가지고 있는가. 시간과 마음을 쓰는 분야, 대상, 활동이다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.domain.profile.merger import normalize_text, taste_rank_score
from app.domain.profile.models import ProfileSignal

KEYWORD_LIMIT = 3
MAX_KEYWORD_LENGTH = 12
# 대상 이름을 뗀 뒤 이보다 짧게 남으면 뗀 값을 쓰지 않는다.
MIN_STRIPPED_LENGTH = 2


@dataclass(slots=True, frozen=True)
class Keyword:
    label: str
    score: float


def _strip_targets(label: str, sources: Sequence[ProfileSignal]) -> str:
    """취향 키워드에 남은 근거 항목의 대상 이름을 뗀다.

    떼고 나서 한 단어만 남으면 "맛있는"처럼 꾸미는 말만 남은 것이라 빈 문자열로 버린다.
    """
    stripped = label
    for target in {(signal.target or "").strip() for signal in sources}:
        if len(target) >= MIN_STRIPPED_LENGTH and target in stripped:
            stripped = " ".join(stripped.replace(target, " ").split())
    if stripped == label:
        return label
    if len(stripped) < MIN_STRIPPED_LENGTH or len(stripped.split()) < 2:
        return ""
    return stripped


def _score(sources: Sequence[ProfileSignal]) -> float:
    return max(taste_rank_score(signal) for signal in sources)


def composed_keywords(
    raw: object,
    signals: Sequence[ProfileSignal],
    *,
    taste: bool,
) -> list[Keyword]:
    """모델이 묶은 키워드 중 근거가 맞는 것만 남긴다.

    raw 는 [{"label": ..., "sources": [id, ...]}] 이고 id 는 signals 의 위치다.
    취향은 근거 항목의 대상 이름이 남아 있으면 떼고, 같은 축의 취향은 하나만 남긴다.
    """
    keywords: list[Keyword] = []
    seen_labels: set[str] = set()
    seen_axes: set[tuple[str, ...]] = set()
    for item in raw if isinstance(raw, list) else []:
        if len(keywords) >= KEYWORD_LIMIT:
            break
        if not isinstance(item, Mapping):
            continue
        label = item.get("label")
        ids = item.get("sources")
        if not isinstance(label, str) or not isinstance(ids, list):
            continue
        sources = [
            signals[index]
            for index in dict.fromkeys(ids)
            if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(signals)
        ]
        if not sources:
            continue
        label = " ".join(label.split())
        if taste:
            label = _strip_targets(label, sources)
        normalized = normalize_text(label)
        if not normalized or len(label) > MAX_KEYWORD_LENGTH or normalized in seen_labels:
            continue
        axes = {signal.taxonomy_path for signal in sources if signal.taxonomy_path}
        if taste and axes & seen_axes:
            continue
        seen_labels.add(normalized)
        seen_axes |= axes
        keywords.append(Keyword(label=label, score=_score(sources)))
    return keywords


def _without_target(signal: ProfileSignal) -> str:
    target = (signal.target or "").strip()
    if not target or target not in signal.value:
        return signal.value
    stripped = " ".join(signal.value.replace(target, " ").split())
    return stripped if len(stripped) >= MIN_STRIPPED_LENGTH else signal.value


def fallback_keywords(signals: Sequence[ProfileSignal], *, taste: bool) -> list[Keyword]:
    """모델 없이 만드는 키워드. signals 는 순위순이다.

    취향은 대상 이름을 떼고 같은 축에서 하나만 쓴다. 관심사는 저장 값 그대로 쓴다.
    """
    keywords: list[Keyword] = []
    seen_labels: set[str] = set()
    seen_axes: set[tuple[str, ...]] = set()
    for signal in signals:
        if len(keywords) >= KEYWORD_LIMIT:
            break
        if taste and signal.taxonomy_path in seen_axes:
            continue
        label = _without_target(signal) if taste else signal.value
        normalized = normalize_text(label)
        if normalized in seen_labels:
            continue
        seen_labels.add(normalized)
        if taste and signal.taxonomy_path:
            seen_axes.add(signal.taxonomy_path)
        keywords.append(Keyword(label=label, score=taste_rank_score(signal)))
    return keywords
