from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LlmTrace:
    """모델 호출을 관측 도구에서 묶는 값. 포트 인자가 아니라 컨텍스트로 넘긴다."""

    session_id: str | None = None
    user_id: str | None = None
    name: str | None = None


_EMPTY = LlmTrace()
_current: ContextVar[LlmTrace | None] = ContextVar("llm_trace", default=None)


def current_llm_trace() -> LlmTrace:
    return _current.get() or _EMPTY


def bind_llm_session(session_id: int, *, user_id: int) -> None:
    """지금 요청의 호출을 대화방과 사용자에 묶는다. 이후 만든 태스크에도 복사된다."""
    _current.set(replace(current_llm_trace(), session_id=str(session_id), user_id=str(user_id)))


@contextmanager
def llm_call(name: str) -> Iterator[None]:
    """블록 안의 모델 호출에 용도 이름을 붙인다."""
    previous = _current.get()
    _current.set(replace(previous or _EMPTY, name=name))
    try:
        yield
    finally:
        # 스트림은 yield를 건너 다른 컨텍스트에서 끝날 수 있어 토큰 reset 대신 값을 되돌린다.
        _current.set(previous)
