"""Small authenticated API over models already loaded in a Colab kernel.

This module is embedded in the downloadable notebook. It performs no model
loading, evaluation or result persistence. JSON schemas guide the prompt and
validate the completed output; they do not constrain the generation logits.
"""

import json
import secrets
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=200000)


class CompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=100)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0, le=2)
    stream: bool = False
    stream_options: dict | None = None
    response_format: dict | None = None


def prepare_messages(body):
    messages = [message.model_dump() for message in body.messages]
    schema = None
    if body.response_format:
        kind = body.response_format.get("type")
        if kind == "json_schema":
            definition = body.response_format.get("json_schema")
            schema = definition.get("schema") if isinstance(definition, dict) else None
            if not isinstance(schema, dict):
                raise HTTPException(422, "json_schema.schema 객체가 필요합니다.")
        elif kind == "json_object":
            schema = {"type": "object"}
        else:
            raise HTTPException(422, "지원하는 response_format은 json_schema와 json_object입니다.")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError:
            raise HTTPException(422, "유효하지 않은 JSON Schema입니다.") from None
        instruction = (
            "\nReturn only valid JSON matching this schema. No markdown or explanation:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        if messages[0]["role"] == "system":
            messages[0]["content"] += instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction.strip()})
    return messages, schema


def create_model_api(model_ids, api_key, generate):
    """generate(body, messages, started) yields strings and a final metadata dict."""
    if not api_key:
        raise ValueError("모델 API 키가 필요합니다.")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {api_key}".encode()):
            return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)
        return await call_next(request)

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "colab"} for model in model_ids],
        }

    @app.post("/v1/chat/completions")
    def completions(body: CompletionRequest):
        started = time.perf_counter()
        if body.model not in model_ids:
            raise HTTPException(404, "로드된 모델 ID를 선택하세요.")
        messages, schema = prepare_messages(body)
        request_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        def events():
            content = ""
            metadata = None
            for event in generate(body, messages, started):
                if isinstance(event, str):
                    content += event
                    if event:
                        yield event
                else:
                    metadata = event
            if metadata is None or not content.strip():
                raise HTTPException(502, "모델이 완전한 응답을 반환하지 않았습니다.")
            if schema is not None:
                try:
                    Draft202012Validator(schema).validate(json.loads(content))
                except (ValueError, ValidationError):
                    raise HTTPException(
                        422, "생성된 JSON이 요청 스키마와 일치하지 않습니다."
                    ) from None
            yield {**metadata, "content": content}

        def chunk(choices, **extra):
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": choices,
                **extra,
            }

        def sse():
            try:
                for event in events():
                    if isinstance(event, str):
                        data = chunk(
                            [{"index": 0, "delta": {"content": event}, "finish_reason": None}]
                        )
                    else:
                        data = chunk(
                            [{"index": 0, "delta": {}, "finish_reason": event["finish_reason"]}],
                            timings=event.get("timings"),
                        )
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if isinstance(event, dict) and (body.stream_options or {}).get("include_usage"):
                        yield f"data: {json.dumps(chunk([], usage=event['usage']))}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as error:
                message = error.detail if isinstance(error, HTTPException) else "모델 생성 실패"
                yield f"data: {json.dumps({'error': {'message': message}}, ensure_ascii=False)}\n\n"

        if body.stream:
            return StreamingResponse(sse(), media_type="text/event-stream")
        try:
            result = None
            for event in events():
                if isinstance(event, dict):
                    result = event
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(502, "모델 생성 실패") from None
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": result["usage"],
            "timings": result.get("timings"),
        }

    return app


def transformers_generator(loaded, max_context_tokens=16384, timeout_seconds=120):
    import queue
    import threading

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    class StopGeneration(StoppingCriteria):
        def __init__(self, cancelled, deadline):
            self.cancelled = cancelled
            self.deadline = deadline

        def __call__(self, input_ids, scores, **kwargs):
            return self.cancelled.is_set() or time.perf_counter() >= self.deadline

    class TimedStreamer(TextIteratorStreamer):
        def __init__(self, tokenizer, started):
            super().__init__(
                tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=timeout_seconds
            )
            self.started = started
            self.first_token_ms = None

        def put(self, value):
            if not self.next_tokens_are_prompt and self.first_token_ms is None:
                self.first_token_ms = (time.perf_counter() - self.started) * 1000
            super().put(value)

    def generate(body, messages, started):
        entry = loaded[body.model]
        tokenizer, model = entry["tokenizer"], entry["model"]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens + body.max_tokens > max_context_tokens:
            raise HTTPException(422, "입력과 최대 출력 토큰이 설정한 컨텍스트 상한을 초과했습니다.")
        if not entry["lock"].acquire(timeout=timeout_seconds):
            raise HTTPException(429, "이 모델의 이전 요청이 아직 실행 중입니다.")
        cancelled = threading.Event()
        deadline = time.perf_counter() + timeout_seconds
        streamer = TimedStreamer(tokenizer, started)
        result = {}
        options = dict(
            **inputs,
            max_new_tokens=body.max_tokens,
            do_sample=body.temperature > 0,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([StopGeneration(cancelled, deadline)]),
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        )
        if body.temperature > 0:
            options["temperature"] = body.temperature

        def run():
            try:
                with torch.inference_mode():
                    output = model.generate(**options)
                result["tokens"] = output.shape[-1] - prompt_tokens
                result["timed_out"] = time.perf_counter() >= deadline
            except Exception:
                result["failed"] = True
                streamer.end()
            finally:
                entry["lock"].release()

        thread = threading.Thread(target=run, daemon=True)
        try:
            thread.start()
        except BaseException:
            entry["lock"].release()
            raise
        try:
            yield from streamer
            thread.join(timeout=timeout_seconds)
            if thread.is_alive() or result.get("timed_out"):
                raise HTTPException(504, "모델 생성 시간 상한을 초과했습니다.")
            if result.get("failed"):
                raise HTTPException(502, "모델 생성 실패")
            count = result["tokens"]
            yield {
                "finish_reason": "length" if count >= body.max_tokens else "stop",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": count,
                    "total_tokens": prompt_tokens + count,
                },
                "timings": {"server_first_token_ms": streamer.first_token_ms},
            }
        except queue.Empty:
            raise HTTPException(504, "모델 스트림 시간 상한을 초과했습니다.") from None
        finally:
            cancelled.set()

    return generate
