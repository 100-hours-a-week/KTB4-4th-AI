"""추천 이유를 상품에 맞는 문장으로 다시 쓴다.

순위를 매길 때 붙은 템플릿 이유는 저장 값을 문형에 끼워 넣어서 "샤워 후 바디로션 바르기와
좋아하는 향 바디로션을 함께 좋아하는"처럼 어색해지기 쉽다. 같은 근거를 모델에 넘겨 상품마다
한 문장으로 다시 쓰고, 모델이 실패하거나 쓴 문장이 기준에 맞지 않으면 템플릿 이유를 그대로 둔다.
근거는 템플릿과 같은 공개 기준(reason_clues)으로 고르므로 선물 목록에 비공개 값이 새지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Mapping, Sequence

from app.application.ports.model_gateway import ModelGateway
from app.application.recommendation_reason_prompts import (
    MAX_REASON_LENGTH,
    MIN_REASON_LENGTH,
    REASON_CHUNK_SIZE,
    ReasonSetPayload,
    build_reason_messages,
)
from app.domain.catalog.labels import CATEGORY_LABELS
from app.domain.catalog.subcategories import SUBCATEGORIES
from app.domain.profile.merger import normalize_text
from app.domain.recommendation import RankedRecommendation, RecommendationMode
from app.domain.recommendation.models import ProductKey
from app.domain.recommendation.reasons import ReasonClues, reason_clues

logger = logging.getLogger(__name__)

# 이모지와 그림 문자. 이유 문장에는 쓰지 않는다.
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️]")


def _category_label(category: str | None, subcategory: str | None) -> str | None:
    if not category:
        return None
    label = CATEGORY_LABELS.get(category)  # type: ignore[call-overload]
    detail = SUBCATEGORIES.get(category, {}).get(subcategory or "")  # type: ignore[call-overload]
    return " ".join(part for part in (label, detail) if part) or None


def valid_reason(raw: object, clues: ReasonClues) -> str | None:
    """모델이 쓴 이유를 쓸 수 있으면 정리해서 돌려준다.

    길이, 이모지, 그리고 이 목록에서 보이면 안 되는 값이 들어갔는지 본다.
    """
    if not isinstance(raw, str):
        return None
    reason = " ".join(raw.split())
    if not MIN_REASON_LENGTH <= len(reason) <= MAX_REASON_LENGTH or _EMOJI.search(reason):
        return None
    normalized = normalize_text(reason)
    if any(
        len(normalize_text(value)) >= 2 and normalize_text(value) in normalized
        for value in clues.hidden
    ):
        return None
    return reason


class RecommendationReasonService:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        model: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._model_gateway = model_gateway
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def rewrite(
        self,
        recommendations: Sequence[RankedRecommendation],
        *,
        recipient_name: str | None = None,
    ) -> dict[ProductKey, str]:
        """다시 쓴 이유. 다시 쓰지 못한 상품은 결과에 없다."""
        if not recommendations:
            return {}
        chunks = [
            recommendations[start : start + REASON_CHUNK_SIZE]
            for start in range(0, len(recommendations), REASON_CHUNK_SIZE)
        ]
        results = await asyncio.gather(
            *(self._rewrite_chunk(chunk, recipient_name=recipient_name) for chunk in chunks)
        )
        return {key: reason for result in results for key, reason in result.items()}

    async def _rewrite_chunk(
        self,
        chunk: Sequence[RankedRecommendation],
        *,
        recipient_name: str | None,
    ) -> dict[ProductKey, str]:
        gift = chunk[0].mode == RecommendationMode.GIFT
        clues = [
            reason_clues(item.matches, item.mode, boost_label=item.boost_label) for item in chunk
        ]
        messages = build_reason_messages(
            [
                (
                    item.product.name,
                    _category_label(item.product.category, item.product.subcategory),
                    clue,
                )
                for item, clue in zip(chunk, clues, strict=True)
            ],
            gift=gift,
            recipient_name=recipient_name if gift else None,
        )
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                raw = await self._model_gateway.structured(
                    messages,
                    model=self._model,
                    json_schema=ReasonSetPayload.model_json_schema(),
                )
        except Exception as error:
            describe = getattr(error, "describe", None)
            logger.warning(
                "reason model failed: mode=%s items=%d model=%s elapsed_ms=%d %s",
                chunk[0].mode.value,
                len(chunk),
                self._model,
                (time.monotonic() - started) * 1000,
                describe() if callable(describe) else f"reason={type(error).__name__}",
            )
            return {}
        rewritten: dict[ProductKey, str] = {}
        answers = raw.get("reasons") if isinstance(raw, Mapping) else None
        for answer in answers if isinstance(answers, list) else []:
            if not isinstance(answer, Mapping):
                continue
            index = answer.get("id")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            if not 0 <= index < len(chunk) or chunk[index].product.key in rewritten:
                continue
            reason = valid_reason(answer.get("reason"), clues[index])
            if reason is not None:
                rewritten[chunk[index].product.key] = reason
        if len(rewritten) < len(chunk):
            logger.info(
                "reason rewrite kept templates: mode=%s kept=%d of %d",
                chunk[0].mode.value,
                len(chunk) - len(rewritten),
                len(chunk),
            )
        return rewritten
