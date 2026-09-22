from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    @property
    def space_id(self) -> str: ...

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...
