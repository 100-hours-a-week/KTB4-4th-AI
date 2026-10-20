"""관심사별 검색 의도를 만들고 캐시한다.

추천 요청(세션 종료)은 사용자가 기다리는 경로라서 거기서 모델을 부르지 않는다. 대화 중에 관심사가
새로 저장되거나 세션을 시작할 때 온보딩 관심사가 들어오면 백그라운드로 의도를 만들어 두고,
추천할 때는 캐시만 읽는다. 의도가 없거나 만들기에 실패한 관심사는 템플릿 쿼리로 찾는다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.application.ports.model_gateway import ModelGateway
from app.application.ports.search_intent_store import SearchIntentStore
from app.application.search_intent_prompts import (
    SEARCH_INTENT_SCHEMA,
    build_search_intent_messages,
    intent_payload,
)
from app.domain.profile.models import TasteField
from app.domain.profile.onboarding import (
    INTEREST_QUERY_TERMS,
    INTEREST_TAXONOMY_KEYS,
    InterestCategory,
)
from app.domain.profile.taxonomy import INTEREST_CATEGORIES_BY_KEY
from app.domain.recommendation import SearchIntent, intent_key, interest_field_name

logger = logging.getLogger(__name__)

# 관심사 쿼리를 만드는 필드. 취향(preferences)은 사용자마다 조합이 달라 의도를 만들지 않는다.
INTENT_FIELDS = frozenset(
    {
        TasteField.INTERESTS,
        TasteField.HOBBIES,
        TasteField.WANTS,
        TasteField.UNAFFORDABLE,
        TasteField.CONSUMABLES,
    }
)


class _Signal(Protocol):
    @property
    def field(self) -> TasteField: ...

    @property
    def value(self) -> str: ...

    @property
    def taxonomy_path(self) -> tuple[str, ...] | None: ...


@dataclass(slots=True, frozen=True)
class IntentTarget:
    value: str
    taxonomy_path: tuple[str, ...] | None = None

    @property
    def key(self) -> str:
        return intent_key(self.value, self.taxonomy_path)


def intent_targets(signals: Iterable[_Signal]) -> list[IntentTarget]:
    """프로필 신호나 추천 신호에서 의도를 만들 관심사만 고른다. 같은 키는 하나만 남긴다."""
    targets: dict[str, IntentTarget] = {}
    for signal in signals:
        if signal.field not in INTENT_FIELDS:
            continue
        target = IntentTarget(signal.value, signal.taxonomy_path)
        targets.setdefault(target.key, target)
    return list(targets.values())


def onboarding_intent_targets(interests: Iterable[InterestCategory]) -> list[IntentTarget]:
    """온보딩 관심 카테고리. 추천 신호와 같은 값과 분야로 만들어야 같은 키가 된다."""
    return [
        IntentTarget(
            INTEREST_QUERY_TERMS[interest],
            INTEREST_CATEGORIES_BY_KEY[INTEREST_TAXONOMY_KEYS[interest]].path,
        )
        for interest in dict.fromkeys(interests)
    ]


class SearchIntentService:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        model: str,
        store: SearchIntentStore,
        timeout_seconds: float = 30.0,
        concurrency: int = 2,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self._gateway = model_gateway
        self._model = model
        self._store = store
        self._timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(concurrency)
        # 여러 턴에서 같은 관심사를 동시에 만들지 않게 진행 중인 키를 둔다.
        self._in_flight: set[str] = set()
        self._tasks: set[asyncio.Task[int]] = set()

    async def lookup(self, targets: Sequence[IntentTarget]) -> Mapping[str, SearchIntent]:
        """캐시에 있는 의도만 돌려준다. 저장소가 실패해도 추천은 템플릿 쿼리로 이어 간다."""
        if not targets:
            return {}
        try:
            return await self._store.get_many([target.key for target in targets])
        except Exception as error:
            logger.warning("search intent lookup failed: reason=%s", type(error).__name__)
            return {}

    async def prepare(self, targets: Iterable[IntentTarget]) -> int:
        """캐시에 없는 의도를 만들어 저장하고, 새로 만든 개수를 돌려준다."""
        pending = {target.key: target for target in targets if target.key not in self._in_flight}
        if not pending:
            return 0
        self._in_flight.update(pending)
        try:
            existing = await self.lookup(list(pending.values()))
            missing = [target for key, target in pending.items() if key not in existing]
            results = await asyncio.gather(*(self._generate(target) for target in missing))
            return sum(results)
        finally:
            self._in_flight.difference_update(pending)

    def schedule(self, targets: Iterable[IntentTarget]) -> None:
        """백그라운드로 prepare 를 돌린다. 대화와 추천 응답은 이 작업을 기다리지 않는다."""
        targets = list(targets)
        if not targets:
            return
        task = asyncio.create_task(self.prepare(targets))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        """서버를 내릴 때 남은 백그라운드 작업을 멈춘다."""
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _generate(self, target: IntentTarget) -> int:
        # 관심사 값은 사용자 발화에서 온 것이라 로그에 남기지 않는다.
        try:
            async with self._semaphore, asyncio.timeout(self._timeout_seconds):
                payload = await self._gateway.structured(
                    build_search_intent_messages(
                        target.value, interest_field_name(target.taxonomy_path)
                    ),
                    model=self._model,
                    json_schema=SEARCH_INTENT_SCHEMA,
                )
            await self._store.save(target.key, SearchIntent.from_payload(intent_payload(payload)))
            return 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "search intent generation failed: model=%s reason=%s",
                self._model,
                getattr(error, "reason", type(error).__name__),
            )
            return 0
