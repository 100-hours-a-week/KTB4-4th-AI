from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

Message = Mapping[str, str]


class ModelGateway(Protocol):
    async def complete(self, messages: Sequence[Message], *, model: str) -> str: ...

    def stream(self, messages: Sequence[Message], *, model: str) -> AsyncIterator[str]: ...

    async def structured(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...
