from collections.abc import Mapping
from typing import Any, Protocol


class JudgmentGateway(Protocol):
    """state와 타입이 정해진 질문(choice, score, noul)을 보내고 질문별 답을 받는다."""

    async def decide(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        model: str,
    ) -> Mapping[str, Any]: ...
