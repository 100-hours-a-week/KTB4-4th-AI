from typing import Protocol


class SessionLease(Protocol):
    """잡은 세션 락. 저장 직전에 renew 로 아직 내 락인지 확인한다."""

    async def renew(self) -> bool:
        """만료 시간을 다시 늘린다. 이미 다른 요청이 가져갔으면 False 다."""
        ...

    async def release(self) -> None: ...


class SessionLock(Protocol):
    """대화 세션 하나에 한 요청만 들어가게 한다. AI 서버가 여러 대여도 같이 본다."""

    async def acquire(self, session_id: int, *, wait_seconds: float) -> SessionLease | None:
        """wait_seconds 동안 기다려도 못 잡으면 None 이다. 0이면 바로 돌려준다."""
        ...

    async def is_locked(self, session_id: int) -> bool: ...
