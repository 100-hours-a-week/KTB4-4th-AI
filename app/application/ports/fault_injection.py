from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ReplyDelayConfig:
    """부하·장애 테스트용 응답 지연 설정. 대상으로 지정한 사용자·세션의 응답 스트림에만 건다."""

    delay_seconds: float
    user_ids: frozenset[int] = frozenset()
    session_ids: frozenset[int] = frozenset()

    def delay_for(self, *, session_id: int, user_id: int) -> float:
        if user_id in self.user_ids or session_id in self.session_ids:
            return max(self.delay_seconds, 0.0)
        return 0.0


class ReplyDelayStore(Protocol):
    """AI 서버가 여러 대여도 같은 설정을 보도록 공유 저장소에 둔다.

    설정에는 만료 시간이 있어서 테스트 뒤에 지우는 것을 잊어도 저절로 풀린다.
    """

    async def get(self) -> ReplyDelayConfig | None: ...

    async def set(self, config: ReplyDelayConfig, *, ttl_seconds: int) -> None: ...

    async def ttl_seconds(self) -> int | None: ...

    async def clear(self) -> None: ...


@dataclass(frozen=True)
class TestAccountConfig:
    """승인된 부하 테스트 계정. 이 계정의 대화는 턴 제한과 자동 마무리를 건너뛴다.

    mock_model이면 모델·외부 API를 부르지 않고 고정 응답을 돌려준다. 끄면 실제 모델(RunPod)을
    부른다. 응답 지연과 따로 둬서, 지연은 장애 구간에만 켜고 계정 정책은 테스트 내내 유지한다.

    mock_opening이면 mock_model이 꺼져 있어도 세션 생성 때 첫인사만 고정 문구로 채운다.
    대화방을 미리 대량으로 만들 때 RunPod에 첫인사 호출이 몰리지 않게 하고, 메시지 응답은
    실제 모델로 부하를 건다.
    """

    __test__ = False

    user_ids: frozenset[int]
    mock_model: bool = True
    mock_opening: bool = False


class TestAccountStore(Protocol):
    async def get(self) -> TestAccountConfig | None: ...

    async def set(self, config: TestAccountConfig, *, ttl_seconds: int) -> None: ...

    async def ttl_seconds(self) -> int | None: ...

    async def clear(self) -> None: ...
