import asyncio
import math
import time

import httpx

from admin_api.client import completion
from benchmarks.fixtures import CASES, SCHEMA, SYSTEM, evaluate


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    left = math.floor(position)
    right = math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def summarize(samples, elapsed, planned):
    good = [s for s in samples if s.get("ok")]
    latencies = [s["latency_ms"] for s in good]
    first = [s["first_chunk_ms"] for s in good if s.get("first_chunk_ms") is not None]
    qualities = [s for s in samples if "expected" in s]
    usage_complete = bool(good) and all(
        isinstance(s.get("usage"), dict)
        and isinstance(s["usage"].get("completion_tokens"), int)
        and not isinstance(s["usage"]["completion_tokens"], bool)
        and s["usage"]["completion_tokens"] >= 0
        for s in good
    )
    output_tokens = sum(s["usage"]["completion_tokens"] for s in good) if usage_complete else None
    return {
        "planned": planned,
        "attempted": len(samples),
        "succeeded": len(good),
        "failed": len(samples) - len(good),
        "complete": len(samples) == planned,
        "success_rate": len(good) / len(samples) if samples else None,
        "elapsed_seconds": elapsed,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "first_chunk_p50_ms": percentile(first, 0.5),
        "first_chunk_p95_ms": percentile(first, 0.95),
        "requests_per_second": len(good) / elapsed if elapsed > 0 else None,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed
        if usage_complete and elapsed > 0
        else None,
        "schema_rate": sum(bool(s.get("evaluation", {}).get("schema_valid")) for s in qualities)
        / len(qualities)
        if qualities
        else None,
        "exact_match_rate": sum(bool(s.get("evaluation", {}).get("exact_match")) for s in qualities)
        / len(qualities)
        if qualities
        else None,
        "truncated": sum(bool(s.get("truncated")) for s in samples),
    }


async def execute(store, run, stop=None, client=None):
    spec = run["spec"]
    if spec["mode"] not in {"quality", "load"}:
        raise ValueError("지원하지 않는 단일 모델 테스트 종류입니다.")
    quality = spec["mode"] == "quality"
    fixture = run.get("manifest", {}).get("fixture", {})
    cases = fixture.get("cases", CASES)
    system = fixture.get("system", SYSTEM)
    schema = fixture.get("schema", SCHEMA)
    planned = len(cases) if quality else spec["requests"]
    started = time.perf_counter()
    queue = asyncio.Queue()
    for ordinal in range(planned):
        queue.put_nowait(ordinal)
    samples = []
    own_client = client is None
    client = client or httpx.AsyncClient(follow_redirects=False)
    tasks = []
    status = "completed"
    pending = {}

    async def consume():
        while not queue.empty():
            if store.cancelled(run["id"]) or (stop and stop.is_set()):
                return
            try:
                ordinal = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            case = cases[ordinal] if quality else None
            prompt = case["input"] if case else spec["prompt"]
            messages = [
                {"role": "system", "content": system if quality else spec["system"]},
                {"role": "user", "content": prompt},
            ]
            pending[ordinal] = {
                "label": case["label"] if case else f"요청 {ordinal + 1}",
                "input": prompt,
            }
            if case:
                pending[ordinal]["expected"] = case["expected"]
            try:
                # httpx read timeout alone doesn't bound a slowly streaming response.
                async with asyncio.timeout(spec["timeout_seconds"]):
                    result = await completion(
                        client, run["connection"], spec, messages, schema if quality else None
                    )
            except TimeoutError:
                result = {"ok": False, "output": "", "error": "요청 전체 시간 상한 도달"}
            sample = {**pending.pop(ordinal), **result}
            if case:
                sample["evaluation"] = evaluate(result.get("output", ""), case["expected"])
                if not result["ok"]:
                    sample["evaluation"].update(schema_valid=False, exact_match=False)
            store.sample(run["id"], ordinal, sample)
            samples.append(sample)

    try:
        tasks = [asyncio.create_task(consume()) for _ in range(min(spec["concurrency"], planned))]
        while not all(t.done() for t in tasks):
            if store.cancelled(run["id"]) or (stop and stop.is_set()):
                status = "cancelled"
                break
            if time.perf_counter() - started >= spec["duration_seconds"]:
                status = "timed_out"
                break
            await asyncio.sleep(0.15)
        if status == "completed":
            await asyncio.gather(*tasks)
            if store.cancelled(run["id"]) or (stop and stop.is_set()):
                status = "cancelled"
            elif not samples or all(not s["ok"] for s in samples):
                status = "failed"
    except Exception:
        status = "failed"
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for ordinal, partial in pending.items():
            sample = {**partial, "ok": False, "output": "", "error": "실행 중단으로 요청 취소"}
            if quality:
                sample["evaluation"] = {"schema_valid": False, "exact_match": False}
            store.sample(run["id"], ordinal, sample)
            samples.append(sample)
        if own_client:
            await client.aclose()
        summary = summarize(samples, time.perf_counter() - started, planned)
        store.finish(run["id"], status, summary)
    return summary
