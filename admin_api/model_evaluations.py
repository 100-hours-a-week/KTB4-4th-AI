"""Observed model results for the lab home; no inferred quality grades."""

import hashlib
import json
from collections import defaultdict

from app.application.conversation_prompts import FRIEND_SUMMARY_SYSTEM
from benchmarks.runner import percentile


def model_evaluations(connections, runs, samples, chats):
    models = {}

    def register(config):
        if not config.get("model"):
            return None
        identity = {
            key: config.get(key) or ""
            for key in ("model", "base_url", "revision", "precision", "hardware")
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        if key not in models:
            models[key] = {
                "id": key,
                **identity,
                "name": config.get("name") or config["model"],
                "connection_ids": [],
                "probe_ok": None,
                "sources": {},
                "calls": defaultdict(list),
                "examples": {},
                "quality": [],
                "last_at": None,
            }
        return models[key]

    for connection in connections:
        model = register(connection)
        if model is not None:
            model["connection_ids"].append(connection["id"])
            if connection.get("probe") is not None:
                model["probe_ok"] = bool(connection["probe"].get("ok"))

    def source(model, kind, record):
        at = record.get("updated_at") or record.get("created_at") or ""
        ref = {
            "kind": kind,
            "id": record["id"],
            "at": at,
            "status": record.get("status"),
        }
        model["sources"][f"{kind}:{record['id']}"] = ref
        model["last_at"] = max(model["last_at"] or "", at)
        return ref

    def observe(config, call, kind, record, user_input="", conversational=True):
        model = register(config)
        if model is None:
            return
        ref = source(model, kind, record)
        # A call saved while generation is in progress has no result yet.
        if not isinstance(call.get("ok"), bool):
            return
        role = call.get("role", "response")
        model["calls"][role].append({**call, "conversational": conversational})
        output = call.get("raw_output") or call.get("output")
        if output is not None and not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        if output or call.get("error"):
            if not user_input:
                user_input = next(
                    (
                        m.get("content", "")
                        for m in reversed(call.get("messages", []))
                        if m.get("role") == "user"
                    ),
                    "",
                )
            example = {
                **ref,
                "input": str(user_input)[:800],
                "output": (output or "")[:1600],
                "ok": call["ok"],
                "error": call.get("error"),
                "latency_ms": call.get("latency_ms"),
            }
            previous = model["examples"].get(role)
            if previous is None or ref["at"] >= previous["at"]:
                model["examples"][role] = example

    samples_by_run = defaultdict(list)
    for sample in samples:
        samples_by_run[sample["run_id"]].append(sample)
    for run in runs:
        pipeline = (run.get("manifest") or {}).get("pipeline")
        if pipeline:
            for role in ("response", "extraction"):
                model = register(pipeline.get(role, {}))
                if model is not None:
                    source(model, "run", run)
        else:
            model = register(run["connection"])
            if model is not None:
                source(model, "run", run)
        for sample in samples_by_run[run["id"]]:
            if pipeline:
                for call in sample.get("model_calls", []):
                    role = call.get("role")
                    if role in {"response", "extraction"}:
                        observe(
                            pipeline[role],
                            call,
                            "run",
                            run,
                            sample.get("input", ""),
                            conversational=sample.get("stage") == "turn",
                        )
            else:
                role = "extraction" if run["spec"]["mode"] == "quality" else "response"
                observe(
                    run["connection"],
                    {**sample, "role": role},
                    "run",
                    run,
                    sample.get("input", ""),
                    conversational=False,
                )
                if role == "extraction" and model is not None and "evaluation" in sample:
                    model["quality"].append(sample["evaluation"])

    for chat in chats:
        for role in ("response", "extraction"):
            config = chat.get("config", {}).get(role, {})
            model = register(config)
            if model is not None:
                source(model, "chat", chat)
        for trace in chat.get("traces", []):
            for call in trace.get("calls", []):
                role = call.get("role")
                if role in {"response", "extraction"}:
                    observe(
                        chat["config"][role],
                        call,
                        "chat",
                        chat,
                        conversational=bool(
                            any(m.get("role") == "user" for m in call.get("messages", []))
                            and not any(
                                m.get("content") == FRIEND_SUMMARY_SYSTEM
                                for m in call.get("messages", [])
                            )
                        ),
                    )

    result = []
    for model in models.values():
        calls = model.pop("calls")
        role_stats = {}
        for role, entries in calls.items():
            good = [call for call in entries if call["ok"]]
            structured = [call for call in entries if call.get("json_schema") is not None]
            role_stats[role] = {
                "attempted": len(entries),
                "failed": len(entries) - len(good),
                "success_rate": len(good) / len(entries),
                "latency_p50_ms": percentile(
                    [c["latency_ms"] for c in good if c.get("latency_ms") is not None], 0.5
                ),
                "latency_p95_ms": percentile(
                    [c["latency_ms"] for c in good if c.get("latency_ms") is not None], 0.95
                ),
                "structured_count": len(structured),
                "structured_success_rate": sum(c["ok"] for c in structured) / len(structured)
                if structured
                else None,
            }
        responses = [
            c["output"]
            for c in calls.get("response", [])
            if c["ok"]
            and c["conversational"]
            and c.get("json_schema") is None
            and isinstance(c.get("output"), str)
            and c["output"].strip()
        ]
        quality = model.pop("quality")
        refs = sorted(model.pop("sources").values(), key=lambda r: r["at"], reverse=True)
        attempted = sum(s["attempted"] for s in role_stats.values())
        failed = sum(s["failed"] for s in role_stats.values())
        result.append(
            {
                **model,
                "roles": role_stats,
                "attempted": attempted,
                "failed": failed,
                "success_rate": (attempted - failed) / attempted if attempted else None,
                "run_count": sum(r["kind"] == "run" for r in refs),
                "chat_count": sum(r["kind"] == "chat" for r in refs),
                "sources": refs[:5],
                "response_style": {
                    "count": len(responses),
                    "median_chars": percentile([len(text) for text in responses], 0.5),
                    "question_rate": sum("?" in t or "？" in t for t in responses) / len(responses)
                    if responses
                    else None,
                },
                "quality": {
                    "count": len(quality),
                    "schema_rate": sum(bool(q.get("schema_valid")) for q in quality) / len(quality)
                    if quality
                    else None,
                    "exact_match_rate": sum(bool(q.get("exact_match")) for q in quality)
                    / len(quality)
                    if quality
                    else None,
                },
            }
        )
    return sorted(result, key=lambda m: (m["attempted"] > 0, m["last_at"] or ""), reverse=True)


def service_evaluations(runs, samples):
    """Query checks belong to the whole service configuration, not one role's model."""
    groups = {}
    for run in runs:
        pipeline = (run.get("manifest") or {}).get("pipeline")
        if not pipeline:
            continue
        settings = {
            role: {
                key: pipeline.get(role, {}).get(key)
                for key in (
                    "model",
                    "base_url",
                    "revision",
                    "precision",
                    "hardware",
                    "max_tokens",
                    "temperature",
                )
            }
            for role in ("response", "extraction", "judgment")
        }
        settings.update(
            judgment_enabled=pipeline.get("judgment_enabled"),
            recommendation_stage=pipeline.get("recommendation_stage"),
        )
        key = json.dumps(settings, sort_keys=True)
        if key not in groups:
            groups[key] = {
                "response_model": pipeline["response"]["model"],
                "extraction_model": pipeline["extraction"]["model"],
                "response_precision": pipeline["response"].get("precision"),
                "extraction_precision": pipeline["extraction"].get("precision"),
                "judgment_enabled": pipeline.get("judgment_enabled"),
                "recommendation_stage": pipeline.get("recommendation_stage"),
                "run_ids": [],
                "completed": 0,
                "reviews": [],
                "turns": [],
            }
        group = groups[key]
        group["run_ids"].append(run["id"])
        group["completed"] += run["status"] == "completed"
        review = (run.get("summary") or {}).get("query_review")
        if run["status"] == "completed" and review and review.get("score") is not None:
            group["reviews"].append(review)
        group["turns"].extend(
            s for s in samples if s["run_id"] == run["id"] and s.get("stage") == "turn"
        )
    result = []
    for group in groups.values():
        reviews, turns = group.pop("reviews"), group.pop("turns")
        conditions = sum(r.get("total_conditions", 0) for r in reviews)
        exclusions = [row for review in reviews for row in review.get("exclusions", [])]
        result.append(
            {
                **group,
                "reviewed_runs": len(reviews),
                "condition_rate": sum(r.get("matched_conditions", 0) for r in reviews) / conditions
                if conditions
                else None,
                "exclusion_rate": sum(row.get("actual") is not None for row in exclusions)
                / len(exclusions)
                if exclusions
                else None,
                "forbidden_count": sum(len(r.get("forbidden_queries", [])) for r in reviews),
                "turn_count": len(turns),
                "extraction_failure_rate": sum(
                    bool(s.get("pipeline", {}).get("extraction_failed")) for s in turns
                )
                / len(turns)
                if turns
                else None,
                "turn_latency_p95_ms": percentile(
                    [
                        s["latency_ms"]
                        for s in turns
                        if s.get("ok") and s.get("latency_ms") is not None
                    ],
                    0.95,
                ),
            }
        )
    return result
