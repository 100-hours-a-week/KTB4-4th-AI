from collections.abc import Mapping
from typing import Any, Protocol


class JobQueue(Protocol):
    async def enqueue(self, *, job_id: str, payload: Mapping[str, Any]) -> None: ...
