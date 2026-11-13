import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest

from app.core.metrics import HTTP_REQUESTS_IN_PROGRESS, PrometheusMiddleware
from tests.unit.test_chat_stream import HEADERS, StreamGateway, make_app, start_session, stream
from tests.unit.test_fault_injection import (
    ACCOUNTS_URL,
    URL,
    ForbiddenGateway,
    sleeps,  # noqa: F401  # fixture
)
from tests.unit.test_fault_injection import make_app as make_fault_app

STREAM_ROUTE = "/v1/chat/sessions/{conversationRoomId}/messages/stream"
SESSIONS_ROUTE = "/v1/chat/sessions"


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def requests_total(route: str, status: str, method: str = "POST") -> float:
    return sample("http_requests_total", method=method, route=route, status=status)


def duration_count(route: str, status: str, method: str = "POST") -> float:
    return sample("http_request_duration_seconds_count", method=method, route=route, status=status)


def test_metrics_endpoint_returns_prometheus_text() -> None:
    with TestClient(make_app(StreamGateway())) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds_bucket" in response.text


def test_requests_are_counted_by_route_template_and_status() -> None:
    before_created = requests_total(SESSIONS_ROUTE, "201")
    before_stream = requests_total(STREAM_ROUTE, "200")
    before_stream_count = duration_count(STREAM_ROUTE, "200")
    with TestClient(make_app(StreamGateway())) as client:
        start_session(client, room_id=4242)
        for user_message_id in (1, 2, 3):
            stream(client, user_message_id, room_id=4242)

    assert requests_total(SESSIONS_ROUTE, "201") == before_created + 1
    assert requests_total(STREAM_ROUTE, "200") == before_stream + 3
    assert duration_count(STREAM_ROUTE, "200") == before_stream_count + 3
    # 경로 파라미터 값(대화방 번호)이 라벨에 들어가지 않는다.
    assert "4242" not in generate_latest().decode()


def test_error_status_is_counted() -> None:
    before = requests_total(SESSIONS_ROUTE, "401")
    with TestClient(make_app(StreamGateway())) as client:
        client.post(SESSIONS_ROUTE, json={"userId": 1, "conversationRoomId": 1})

    assert requests_total(SESSIONS_ROUTE, "401") == before + 1


def test_health_and_metrics_are_not_counted() -> None:
    with TestClient(make_app(StreamGateway())) as client:
        client.get("/health")
        client.get("/metrics")
        client.get("/openapi.json")

    text = generate_latest().decode()
    for path in ("/health", "/metrics", "/openapi.json"):
        assert f'route="{path}"' not in text


def test_unknown_paths_share_one_route_label() -> None:
    before = requests_total("unmatched", "404", method="GET")
    with TestClient(make_app(StreamGateway())) as client:
        client.get("/wp-admin/12345")
        client.get("/random/67890")

    assert requests_total("unmatched", "404", method="GET") == before + 2
    assert "67890" not in generate_latest().decode()


def test_in_progress_gauge_rises_during_request_and_falls_after() -> None:
    app = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: int) -> dict[str, float]:
        gauge = HTTP_REQUESTS_IN_PROGRESS.labels(method="GET")
        return {"during": gauge._value.get()}

    app.add_middleware(PrometheusMiddleware)
    with TestClient(app) as client:
        during = client.get("/items/7").json()["during"]

    assert during >= 1
    assert sample("http_requests_in_progress", method="GET") == 0
    assert requests_total("/items/{item_id}", "200", method="GET") >= 1


def test_histogram_buckets_cover_long_replies() -> None:
    with TestClient(make_app(StreamGateway())) as client:
        start_session(client, room_id=5151)
        stream(client, 1, room_id=5151)
        text = client.get("/metrics").text

    for bucket in ("0.1", "3.0", "120.0", "+Inf"):
        assert f'le="{bucket}",method="POST",route="{STREAM_ROUTE}"' in text


@pytest.mark.usefixtures("sleeps")
def test_mock_reply_and_delay_are_counted() -> None:
    before_mock = sample("needu_mock_replies_total", kind="reply")
    before_opening = sample("needu_mock_replies_total", kind="opening")
    before_delay = sample("needu_fault_reply_delay_total")
    before_delay_seconds = sample("needu_fault_reply_delay_seconds_total")
    with TestClient(make_fault_app(gateway=ForbiddenGateway())) as client:
        client.put(ACCOUNTS_URL, headers=HEADERS, json={"userIds": [1]})
        client.put(URL, headers=HEADERS, json={"delaySeconds": 3, "userIds": [1]})
        start_session(client)
        stream(client, 1)
        stream(client, 2)

    assert sample("needu_mock_replies_total", kind="opening") == before_opening + 1
    assert sample("needu_mock_replies_total", kind="reply") == before_mock + 2
    assert sample("needu_fault_reply_delay_total") == before_delay + 2
    assert sample("needu_fault_reply_delay_seconds_total") == before_delay_seconds + 6
