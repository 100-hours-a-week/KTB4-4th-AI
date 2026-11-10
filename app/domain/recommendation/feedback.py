"""추천 상품에 남긴 피드백을 다음 추천에 반영한다.

Backend는 추천 때 어떤 취향이 그 상품을 골랐는지 저장하지 않는다. 그래서 추천할 때마다 이번 검색
결과에서 피드백 상품이 가장 가깝게 걸린 취향을 다시 찾아 그 취향의 가중치를 조정한다.
이번 검색 결과에 걸리지 않은 피드백 상품은 상품 제외에만 쓴다.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from app.domain.recommendation.intents import intent_key
from app.domain.recommendation.models import ProductKey, SearchQuery, VectorMatch


class FeedbackAction(StrEnum):
    LIKE = "LIKE"
    DISLIKE = "DISLIKE"
    # 사용자가 추천 링크로 들어가 샀다고 답했다.
    PURCHASED = "PURCHASED"
    # 친구가 이 사용자의 선물 목록에서 골라 샀다고 답했다. 곧 가지게 될 물건이다.
    RECEIVED = "RECEIVED"


# 피드백 하나가 그 상품을 고른 취향의 가중치를 바꾸는 폭. 싫어요는 좋아요보다 크게 본다.
# 산 것은 그 취향이 맞았다는 뜻이라 좋아요만큼 올린다.
SIGNAL_STEP = {
    FeedbackAction.LIKE: 0.05,
    FeedbackAction.DISLIKE: -0.10,
    FeedbackAction.PURCHASED: 0.05,
    FeedbackAction.RECEIVED: 0.05,
}
# 피드백이 많이 쌓여도 대화에서 들은 취향을 뒤집지 않게 좁게 묶는다.
MIN_SIGNAL_WEIGHT = 0.6
MAX_SIGNAL_WEIGHT = 1.3
# 이미 가졌거나 곧 가질 상품과 세부 카테고리가 같은 후보의 점수 배율. 같은 물건을 또 권하지 않는다.
OWNED_SIMILAR_WEIGHT = 0.85
_OWNED_ACTIONS = frozenset({FeedbackAction.PURCHASED, FeedbackAction.RECEIVED})


@dataclass(slots=True, frozen=True)
class ProductFeedback:
    key: ProductKey
    action: FeedbackAction


@dataclass(slots=True, frozen=True)
class FeedbackSummary:
    """상품마다 하나로 정리한 피드백. 같은 상품에 여러 피드백이 있으면 싫어요와 구매를 앞세운다."""

    actions: Mapping[ProductKey, FeedbackAction]

    @classmethod
    def of(cls, events: Iterable[ProductFeedback]) -> FeedbackSummary:
        priority = {
            FeedbackAction.LIKE: 0,
            FeedbackAction.RECEIVED: 1,
            FeedbackAction.PURCHASED: 2,
            FeedbackAction.DISLIKE: 3,
        }
        actions: dict[ProductKey, FeedbackAction] = {}
        for event in events:
            current = actions.get(event.key)
            if current is None or priority[event.action] > priority[current]:
                actions[event.key] = event.action
        return cls(actions=actions)

    @property
    def excluded(self) -> frozenset[ProductKey]:
        """다시 추천하지 않을 상품. 싫다고 했거나 이미 가진 상품이다."""
        return frozenset(
            key for key, action in self.actions.items() if action != FeedbackAction.LIKE
        )


def _signal_key(query: SearchQuery) -> str:
    return intent_key(query.signal.value, query.signal.taxonomy_path)


def signal_weights(
    matches: Iterable[VectorMatch],
    feedback: FeedbackSummary,
) -> dict[str, float]:
    """쿼리(query_id)마다 곱할 피드백 가중치. 1.0 인 쿼리는 넣지 않는다.

    피드백 상품을 가장 가깝게 찾은 쿼리의 취향에 피드백을 돌린다. 같은 취향에서 나온
    다른 쿼리(content·usage·gift)도 같은 가중치를 받는다.
    """
    matches = list(matches)
    nearest: dict[ProductKey, VectorMatch] = {}
    for match in matches:
        key = match.product.key
        if key in feedback.actions and (
            key not in nearest or match.similarity > nearest[key].similarity
        ):
            nearest[key] = match
    totals: dict[str, float] = defaultdict(float)
    for key, match in nearest.items():
        totals[_signal_key(match.query)] += SIGNAL_STEP[feedback.actions[key]]
    weights: dict[str, float] = {}
    for match in matches:
        total = totals.get(_signal_key(match.query))
        if total:
            weights[match.query.query_id] = min(
                max(1.0 + total, MIN_SIGNAL_WEIGHT), MAX_SIGNAL_WEIGHT
            )
    return weights


def owned_similar_weights(
    matches: Iterable[VectorMatch],
    feedback: FeedbackSummary,
) -> dict[ProductKey, float]:
    """이미 가졌거나 곧 가질 상품과 세부 카테고리가 같은 후보의 점수 배율."""
    matches = list(matches)
    owned = {
        match.product.subcategory
        for match in matches
        if feedback.actions.get(match.product.key) in _OWNED_ACTIONS and match.product.subcategory
    }
    return {
        match.product.key: OWNED_SIMILAR_WEIGHT
        for match in matches
        if match.product.subcategory in owned and match.product.key not in feedback.actions
    }
