import json
import time

import httpx

from admin_api.credentials import connection_key


def headers(connection):
    key = connection_key(connection)
    return {"Authorization": f"Bearer {key}"} if key else {}


def huggingface_models(rows):
    metadata = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        for provider in row.get("providers", []):
            if (
                not isinstance(provider, dict)
                or provider.get("status") != "live"
                or not isinstance(provider.get("provider"), str)
            ):
                continue
            metadata.append(
                {
                    "id": f"{row['id']}:{provider['provider']}",
                    "model": row["id"],
                    "provider": provider["provider"],
                    "supports_structured_output": provider.get("supports_structured_output"),
                    "context_length": provider.get("context_length"),
                }
            )
    return metadata


def resolve_huggingface_model(model, metadata):
    requested = model.strip()
    exact = next((item for item in metadata if item["id"] == requested), None)
    if exact:
        return exact
    candidates = [item for item in metadata if item["model"] == requested]
    if not candidates:
        raise ValueError(
            "이 모델 또는 제공업체는 현재 Hugging Face Chat API 목록에 없습니다. "
            "Hub에 가중치가 있어도 API로 바로 호출할 수 있는 것은 아닙니다. "
            "목록의 모델을 선택하거나 별도 전용 엔드포인트를 배포하세요."
        )
    # Prefer a provider advertising structured output; record the resolved route explicitly.
    return next(
        (item for item in candidates if item.get("supports_structured_output") is True),
        candidates[0],
    )


async def probe(client, connection):
    started = time.perf_counter()
    try:
        credential_error = None
        try:
            request_headers = headers(connection)
        except ValueError as error:
            if connection.get("provider") != "huggingface":
                raise
            # The router's model catalogue is public; listing it doesn't verify inference access.
            credential_error = error_message(error)
            request_headers = {}
        if connection.get("provider") == "huggingface" and not request_headers:
            credential_error = credential_error or "Hugging Face 추론 토큰을 설정하세요."
        response = await client.get(
            f"{connection['base_url']}/models", headers=request_headers, timeout=10
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("data")
        if not isinstance(rows, list):
            raise ValueError("/models 응답이 OpenAI 호환 형식이 아닙니다.")
        models = [
            row["id"] for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)
        ]
        result = {
            "ok": not credential_error,
            "models": models,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        if connection.get("provider") == "huggingface":
            metadata = huggingface_models(rows)
            result.update(
                models=[entry["id"] for entry in metadata],
                model_metadata=metadata,
                credential_configured=bool(request_headers),
                inference_verified=False,
            )
        if credential_error:
            result["error"] = credential_error + " 모델 목록만 조회했습니다."
        return result
    except (httpx.HTTPError, ValueError, TypeError) as error:
        return {"ok": False, "models": [], "error": error_message(error)}


def error_message(error):
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code == 401:
            return "모델 서버 HTTP 401. API 토큰이 유효한지 확인하세요."
        if error.response.status_code == 403:
            return "모델 서버 HTTP 403. 토큰의 추론 권한과 모델 접근 권한을 확인하세요."
        if error.response.status_code == 402:
            return "모델 서버 HTTP 402. 추론 크레딧과 결제 설정을 확인하세요."
        return f"모델 서버 HTTP {error.response.status_code}. API 경로·모델·옵션 지원을 확인하세요."
    if isinstance(error, httpx.TimeoutException):
        return "모델 요청 시간 초과"
    if isinstance(error, httpx.RequestError):
        return "모델 서버에 연결할 수 없습니다. 서버 실행과 API URL을 확인하세요."
    if isinstance(error, ValueError):
        return (
            str(error)
            if not isinstance(error, json.JSONDecodeError)
            else "모델 응답 JSON 파싱 실패"
        )
    return f"모델 응답 처리 실패 ({type(error).__name__})"


async def completion(client, connection, spec, messages, schema=None):
    payload = {
        "model": connection["model"],
        "messages": messages,
        "temperature": spec["temperature"],
        "max_tokens": spec["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "taste_smoke", "strict": True, "schema": schema},
        }
    started = time.perf_counter()
    content = ""
    first_chunk_ms = None
    usage = None
    finish_reason = None
    done = False
    try:
        async with client.stream(
            "POST",
            f"{connection['base_url']}/chat/completions",
            headers=headers(connection),
            json=payload,
            timeout=httpx.Timeout(spec["timeout_seconds"]),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                if not data:
                    continue
                body = json.loads(data)
                if body.get("error"):
                    raise ValueError("모델 스트림이 오류를 반환했습니다.")
                if isinstance(body.get("usage"), dict):
                    usage = body["usage"]
                choices = body.get("choices", [])
                if choices:
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    text = choice.get("delta", {}).get("content")
                    if isinstance(text, str) and text:
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - started) * 1000
                        content += text
                        if len(content) > 50000:
                            raise ValueError("모델 출력이 저장 상한을 초과했습니다.")
        if not done and not finish_reason:
            raise ValueError("모델 스트림이 완료 신호 없이 끊겼습니다.")
        if not content.strip():
            raise ValueError("모델이 비어 있는 응답을 반환했습니다.")
        return {
            "ok": True,
            "output": content,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "first_chunk_ms": first_chunk_ms,
            "usage": usage,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        return {
            "ok": False,
            "output": content,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "first_chunk_ms": first_chunk_ms,
            "usage": usage,
            "error": error_message(error),
        }
