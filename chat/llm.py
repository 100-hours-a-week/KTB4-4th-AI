"""LM Studio 클라이언트. 의존성 없음(stdlib)."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent


def _load_env() -> None:
    """.env 를 읽되 실제 환경변수가 항상 이긴다."""
    for name in (".env", ".env.example"):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
        break


_load_env()

BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

RESPONSE_MODEL = os.environ.get("NIJU_RESPONSE_MODEL", "kanana-2-3b-instruct")
EXTRACT_MODEL = os.environ.get("NIJU_EXTRACT_MODEL", RESPONSE_MODEL)

RESPONSE_TEMP = float(os.environ.get("NIJU_RESPONSE_TEMP", "0.7"))
EXTRACT_TEMP = float(os.environ.get("NIJU_EXTRACT_TEMP", "0.1"))

# reasoning 모델(qwen3 등)은 thinking 이 켜져 있으면 첫 글자가 수십 초 뒤에 나온다.
NO_THINK = os.environ.get("NIJU_NO_THINK", "0") not in ("0", "", "false")


def _maybe_no_think(body: dict) -> dict:
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def _post(body: dict, stream: bool):
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    return urllib.request.urlopen(req, timeout=600)


# 모델이 종료 토큰을 본문으로 흘리는 경우가 있다 (kanana 의 <|eot_id|> 등)
_LEAK = ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>",
         "<|start_header_id|>", "<|end_header_id|>", "<|channel|>", "<|message|>")


def _clean(text: str) -> str:
    for t in _LEAK:
        text = text.replace(t, "")
    return text


_META = re.compile(r"(\[\s*지시\s*\]|지시\s*[:：]|이번 턴에 할 일|내부 지시|"
                   r"\[\s*내부\s*\]|사용자는 볼 수 없다).*", re.S)


def strip_meta(text: str) -> str:
    """지시문이 답변에 섞여 나오면 그 지점부터 잘라낸다."""
    return _META.sub("", text).strip()


def one_question(text: str) -> str:
    """물음표가 둘 이상이면 첫 질문까지만 남긴다. '질문은 하나' 규칙을 코드로 강제한다."""
    if text.count("?") + text.count("？") < 2:
        return text
    out, seen = [], False
    for part in re.split(r"(?<=[.!?？。])\s+", text):
        out.append(part)
        if "?" in part or "？" in part:
            seen = True
            break
    return " ".join(out).strip() if seen else text


def stream_chat(messages: list[dict], model: str = "", temperature: float | None = None) -> Iterator[str]:
    """응답 모델. 토큰을 흘려보낸다."""
    body = {
        "model": model or RESPONSE_MODEL,
        "messages": messages,
        "temperature": RESPONSE_TEMP if temperature is None else temperature,
        "max_tokens": 400,
        "stream": True,
    }
    with _post(_maybe_no_think(body), stream=True) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield _clean(piece)


def _first_json(text: str) -> str:
    """텍스트에서 첫 JSON 객체만 잘라낸다. 앞뒤에 설명이 붙어 나와도 살린다."""
    text = text.strip()
    if not text:
        return "{}"
    start = text.find("{")
    if start < 0:
        return "{}"
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def json_chat(messages: list[dict], schema: dict, model: str = "", temperature: float | None = None) -> dict:
    """추출 모델. 구조화 출력으로 형식을 강제한다."""
    body = {
        "model": model or EXTRACT_MODEL,
        "messages": messages,
        "temperature": EXTRACT_TEMP if temperature is None else temperature,
        "max_tokens": 800,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "delta", "strict": True, "schema": schema},
        },
    }
    last = None
    for attempt in range(3):
        try:
            with _post(_maybe_no_think(body), stream=False) as r:
                data = json.loads(r.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            raw = msg.get("content") or ""
            if not raw.strip():
                # reasoning 모델은 답을 reasoning 채널로 흘려보내고 content 를 비우는 일이 있다.
                # LM Studio + qwen3 조합에서 재현된다. 버리지 말고 여기서 건져낸다.
                raw = msg.get("reasoning") or msg.get("reasoning_content") or ""
            return json.loads(_first_json(raw))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"추출 호출 실패({BASE_URL}): {last}")
