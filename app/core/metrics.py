"""Prometheus 메트릭.

HTTP 요청 수·처리 시간·처리 중인 요청 수와, 부하 테스트용 장애 주입(Mock 응답, 응답 지연)이
실제로 걸렸는지 확인하는 카운터를 둔다. 서버는 Pod당 uvicorn 워커 하나라 기본 레지스트리를 쓴다.
"""

from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# 응답 성능 통계에서 뺄 내부 경로. Prometheus 수집과 헬스체크가 AI 요청 통계를 늘리지 않게 한다.
EXCLUDED_PATHS = frozenset(
    {"/health", "/metrics", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
)
# 매칭되는 라우트가 없는 요청(스캐너 등)은 경로를 그대로 쓰지 않고 하나로 묶는다.
UNMATCHED_ROUTE = "unmatched"
# 대화 응답은 수 초, 장애 주입·적체 때는 수십 초 이상 걸릴 수 있어 넓게 잡는다.
DURATION_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60, 90, 120)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "AI FastAPI HTTP 요청 수",
    ("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "AI FastAPI HTTP 요청 처리 시간. SSE는 스트림을 끝까지 보낸 시점까지 잰다.",
    ("method", "route", "status"),
    buckets=DURATION_BUCKETS,
)
# 라우트는 라우팅이 끝나야 알 수 있어서 요청을 받는 시점에 올리는 이 지표에는 method만 단다.
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "AI FastAPI에서 처리 중인 HTTP 요청 수",
    ("method",),
)

FAULT_REPLY_DELAYS = Counter(
    "needu_fault_reply_delay_total",
    "부하 테스트 응답 지연을 건 횟수",
)
FAULT_REPLY_DELAY_SECONDS = Counter(
    "needu_fault_reply_delay_seconds_total",
    "부하 테스트 응답 지연으로 기다린 시간 합계",
)
MOCK_REPLIES = Counter(
    "needu_mock_replies_total",
    "부하 테스트 계정에 모델 대신 돌려준 Mock 응답 수",
    ("kind",),
)


def record_reply_delay(seconds: float) -> None:
    FAULT_REPLY_DELAYS.inc()
    FAULT_REPLY_DELAY_SECONDS.inc(seconds)


def record_mock_reply(kind: str) -> None:
    MOCK_REPLIES.labels(kind=kind).inc()


def _route_template(scope: Scope) -> str:
    """매칭된 라우트의 경로 템플릿. 경로 파라미터 값이 라벨에 들어가지 않게 한다.

    FastAPI는 하위 라우터 기준 상대 경로만 scope["route"]에 남기므로, 실제 경로에서 경로
    파라미터 값이 들어간 마디만 {이름}으로 바꾼다. 매칭되지 않은 요청(404 등)은 하나로 묶는다.
    """
    if scope.get("route") is None:
        return UNMATCHED_ROUTE
    names = {str(value): name for name, value in scope.get("path_params", {}).items()}
    return "/".join(
        f"{{{names[segment]}}}" if segment in names else segment
        for segment in scope["path"].split("/")
    )


class PrometheusMiddleware:
    """HTTP 요청 수·처리 시간·처리 중인 요청 수를 기록한다.

    BaseHTTPMiddleware는 응답 헤더를 받은 시점에 끝나서 SSE 시간을 못 잰다. 그래서 ASGI로 감싸
    응답 본문을 끝까지 보낸 뒤(또는 클라이언트가 끊은 뒤) 기록한다.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in EXCLUDED_PATHS:
            await self._app(scope, receive, send)
            return

        method = scope["method"]
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        in_progress = HTTP_REQUESTS_IN_PROGRESS.labels(method=method)
        in_progress.inc()
        started = time.perf_counter()
        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            in_progress.dec()
            labels = {"method": method, "route": _route_template(scope), "status": str(status)}
            HTTP_REQUESTS.labels(**labels).inc()
            HTTP_REQUEST_DURATION.labels(**labels).observe(elapsed)


async def metrics_endpoint(request: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
