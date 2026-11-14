from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request, Response, status
from pydantic import Field

from app.api.schemas.common import ApiModel
from app.application.ports.fault_injection import (
    ReplyDelayConfig,
    ReplyDelayStore,
    TestAccountConfig,
    TestAccountStore,
)
from app.core.errors import ApiError

router = APIRouter()


class ReplyDelayRequest(ApiModel):
    delay_seconds: float = Field(gt=0, le=60)
    user_ids: list[int] = Field(default_factory=list, max_length=10000)
    session_ids: list[int] = Field(default_factory=list, max_length=10000)
    # 지우는 것을 잊어도 풀리도록 만료를 반드시 건다. 기본 1시간, 최대 6시간.
    ttl_seconds: int = Field(default=3600, ge=60, le=21600)


class ReplyDelayResponse(ApiModel):
    active: bool
    delay_seconds: float = 0.0
    max_delay_seconds: float
    user_ids: list[int] = Field(default_factory=list)
    session_ids: list[int] = Field(default_factory=list)
    expires_in_seconds: int | None = None


def _store(request: Request) -> ReplyDelayStore:
    store = getattr(request.app.state, "reply_delay_store", None)
    if store is None:
        raise ApiError(
            status_code=404,
            code="FAULT_INJECTION_DISABLED",
            message="이 서버는 장애 주입이 꺼져 있습니다. FAULT_INJECTION_ENABLED를 확인하세요.",
            retryable=False,
        )
    return cast(ReplyDelayStore, store)


async def _current(request: Request, store: ReplyDelayStore) -> ReplyDelayResponse:
    max_delay = request.app.state.settings.fault_injection_max_delay_seconds
    config = await store.get()
    if config is None:
        return ReplyDelayResponse(active=False, max_delay_seconds=max_delay)
    return ReplyDelayResponse(
        active=True,
        delay_seconds=min(config.delay_seconds, max_delay),
        max_delay_seconds=max_delay,
        user_ids=sorted(config.user_ids),
        session_ids=sorted(config.session_ids),
        expires_in_seconds=await store.ttl_seconds(),
    )


@router.get("/reply-delay", response_model=ReplyDelayResponse, response_model_by_alias=True)
async def get_reply_delay(request: Request) -> ReplyDelayResponse:
    return await _current(request, _store(request))


@router.put("/reply-delay", response_model=ReplyDelayResponse, response_model_by_alias=True)
async def set_reply_delay(body: ReplyDelayRequest, request: Request) -> ReplyDelayResponse:
    store = _store(request)
    # 대상 없이 켜면 아무 요청에도 걸리지 않아 테스트가 헛돈다. 전체 적용은 일부러 막는다.
    if not body.user_ids and not body.session_ids:
        raise ApiError(
            status_code=422,
            code="FAULT_TARGET_REQUIRED",
            message="userIds 또는 sessionIds 중 하나는 있어야 합니다.",
        )
    await store.set(
        ReplyDelayConfig(
            delay_seconds=body.delay_seconds,
            user_ids=frozenset(body.user_ids),
            session_ids=frozenset(body.session_ids),
        ),
        ttl_seconds=body.ttl_seconds,
    )
    return await _current(request, store)


@router.delete("/reply-delay", status_code=status.HTTP_204_NO_CONTENT)
async def clear_reply_delay(request: Request) -> Response:
    await _store(request).clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class TestAccountsRequest(ApiModel):
    __test__ = False

    user_ids: list[int] = Field(min_length=1, max_length=100000)
    # true면 모델·외부 API 대신 고정 응답을 쓴다. false면 실제 모델(RunPod)을 부른다.
    mock_model: bool = True
    # true면 mockModel이 false여도 세션 생성 때 첫인사만 고정 문구로 채운다(모델 호출 없음).
    mock_opening: bool = False
    # 테스트 내내 유지하도록 기본 2시간, 최대 12시간.
    # 이미 정책이 켜진 세션은 목록이 만료돼도 그대로 유지된다.
    ttl_seconds: int = Field(default=7200, ge=60, le=43200)


class TestAccountsResponse(ApiModel):
    __test__ = False

    active: bool
    user_count: int
    mock_model: bool | None = None
    mock_opening: bool | None = None
    expires_in_seconds: int | None = None


def _accounts(request: Request) -> TestAccountStore:
    store = getattr(request.app.state, "test_account_store", None)
    if store is None:
        raise ApiError(
            status_code=404,
            code="FAULT_INJECTION_DISABLED",
            message="이 서버는 장애 주입이 꺼져 있습니다. FAULT_INJECTION_ENABLED를 확인하세요.",
            retryable=False,
        )
    return cast(TestAccountStore, store)


async def _current_accounts(store: TestAccountStore) -> TestAccountsResponse:
    config = await store.get()
    if config is None or not config.user_ids:
        return TestAccountsResponse(active=False, user_count=0)
    return TestAccountsResponse(
        active=True,
        user_count=len(config.user_ids),
        mock_model=config.mock_model,
        mock_opening=config.mock_opening,
        expires_in_seconds=await store.ttl_seconds(),
    )


@router.get("/test-accounts", response_model=TestAccountsResponse, response_model_by_alias=True)
async def get_test_accounts(request: Request) -> TestAccountsResponse:
    return await _current_accounts(_accounts(request))


@router.put("/test-accounts", response_model=TestAccountsResponse, response_model_by_alias=True)
async def set_test_accounts(body: TestAccountsRequest, request: Request) -> TestAccountsResponse:
    store = _accounts(request)
    await store.set(
        TestAccountConfig(
            user_ids=frozenset(body.user_ids),
            mock_model=body.mock_model,
            mock_opening=body.mock_opening,
        ),
        ttl_seconds=body.ttl_seconds,
    )
    return await _current_accounts(store)


@router.delete("/test-accounts", status_code=status.HTTP_204_NO_CONTENT)
async def clear_test_accounts(request: Request) -> Response:
    await _accounts(request).clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
