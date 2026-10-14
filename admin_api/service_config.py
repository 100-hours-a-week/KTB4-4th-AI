"""Expose service configuration without exposing credentials or mutating .env."""

import hashlib
import json
from urllib.parse import urlsplit

from admin_api.credentials import connection_key
from admin_api.store import ROOT
from app.application.conversation_prompts import (
    CANDIDATE_SYSTEM,
    EXTRACTION_SYSTEM,
    build_opening_messages,
)
from app.application.conversation_service import ExtractionDeltaPayload
from app.application.taste_judgment import CandidateDeltaPayload
from app.core.config import Settings


def load_service_settings():
    # Explicitly authorized for service-mode runs; no Redis/MySQL session access is opened.
    return Settings(_env_file=ROOT / ".env")


def public_url(value):
    if value is None:
        return None
    parts = urlsplit(str(value))
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}{parts.path}".rstrip("/")


def environment_snapshot(settings):
    return {
        "response": {
            "base_url": public_url(settings.response_model_base_url),
            "model": settings.response_model_name,
            "max_tokens": settings.response_model_max_tokens,
            "timeout_seconds": settings.response_model_timeout_seconds,
        },
        "extraction": {
            "base_url": public_url(settings.extraction_model_base_url),
            "model": settings.extraction_model_name,
            "max_tokens": settings.extraction_model_max_tokens,
            "timeout_seconds": settings.extraction_model_timeout_seconds,
        },
        "judgment": {
            "base_url": public_url(settings.judgment_model_base_url),
            "model": settings.judgment_model_name,
            "timeout_seconds": settings.judgment_model_timeout_seconds,
        },
        "embedding": {
            "base_url": public_url(settings.embedding_base_url),
            "model": settings.embedding_query_model_name,
            "space_id": settings.embedding_space_id,
            "dimensions": settings.embedding_dimensions,
            "batch_size": settings.embedding_batch_size,
            "timeout_seconds": settings.embedding_timeout_seconds,
        },
        "summary_timeout_seconds": settings.summary_model_timeout_seconds,
        "persona": persona_configuration(settings),
        "session_idle_ttl_seconds": settings.chat_session_idle_ttl_seconds,
        "session_max_lifetime_seconds": settings.chat_session_max_lifetime_seconds,
        "catalog_configured": bool(
            settings.catalog_database_url_ro or settings.catalog_database_url
        ),
        # Never expose a DB DSN. The hash below only includes its non-credential destination.
        "catalog_destination": public_url(
            settings.catalog_database_url_ro or settings.catalog_database_url
        ),
    }


def persona_configuration(settings):
    if settings.persona_model_base_url and settings.persona_model_name:
        base_url, model = settings.persona_model_base_url, settings.persona_model_name
    else:
        base_url, model = None, None
        for role in ("extraction", "response"):
            candidate = getattr(settings, f"{role}_model_name")
            endpoint = getattr(settings, f"{role}_model_base_url")
            if endpoint and "gemini" in candidate.lower():
                base_url, model = endpoint, candidate
                break
    return {
        "base_url": public_url(base_url),
        "model": model,
        "credential_source": "connection" if settings.persona_model_api_key_env else "service",
        "api_key_env": settings.persona_model_api_key_env,
        "max_tokens": 768,
        "timeout_seconds": 60,
        "temperature": 0.7,
    }


def environment_hash(settings):
    return hashlib.sha256(
        json.dumps(environment_snapshot(settings), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def service_configuration(settings):
    snapshot = environment_snapshot(settings)
    snapshot.pop("catalog_destination")
    snapshot["source"] = ".env + 프로세스 환경변수 (환경변수 우선)"
    snapshot["opening_messages"] = build_opening_messages()
    snapshot["extraction_prompts"] = {
        "direct": {
            "system": EXTRACTION_SYSTEM,
            "schema": ExtractionDeltaPayload.model_json_schema(by_alias=True),
        },
        "judgment": {
            "system": CANDIDATE_SYSTEM,
            "schema": CandidateDeltaPayload.model_json_schema(),
        },
    }
    snapshot["credential_status"] = {
        "model_key_configured": bool(settings.model_api_key or settings.openrouter_api_key),
        "embedding_key_configured": bool(settings.upstage_api_key),
        "huggingface_key_configured": bool(settings.hf_token),
    }
    return snapshot


def pipeline_configuration(settings, spec, store):
    snapshot = environment_snapshot(settings)
    snapshot.pop("catalog_destination")
    for role, connection_id in [
        ("response", spec.response_connection_id),
        ("extraction", spec.extraction_connection_id),
    ]:
        role_config = snapshot[role]
        role_config["credential_source"] = "service"
        role_config["name"] = "환경변수 기본 모델"
        if connection_id:
            saved = store.connection(connection_id)
            if not saved or not saved["model"].strip():
                raise ValueError(f"{role} 역할의 연결과 실제 모델 ID를 확인하세요.")
            role_config.update(
                **{
                    key: saved[key]
                    for key in ["name", "base_url", "model", "revision", "precision", "hardware"]
                },
                credential_source="connection",
                api_key_env=saved["api_key_env"],
                provider=saved.get("provider"),
            )
            if saved.get("provider") == "huggingface":
                metadata = next(
                    (
                        item
                        for item in (saved.get("probe") or {}).get("model_metadata", [])
                        if item["id"] == saved["model"]
                    ),
                    None,
                )
                role_config["hosting"] = metadata or {
                    "id": saved["model"],
                    "capabilities_verified": False,
                }
                if (
                    role == "extraction"
                    and metadata
                    and metadata.get("supports_structured_output") is False
                ):
                    raise ValueError(
                        "선택한 Hugging Face 제공업체는 구조화 추출을 지원하지 않습니다. "
                        "JSON Schema 지원 제공업체를 선택하세요."
                    )
        if not role_config["base_url"] or not role_config["model"]:
            raise ValueError(f"{role} 모델의 환경변수를 설정하거나 등록된 연결을 선택하세요.")
    if spec.judgment_mode == "disabled":
        snapshot["judgment"]["base_url"] = None
    snapshot["judgment_enabled"] = bool(snapshot["judgment"]["base_url"])
    snapshot["recommendation_stage"] = spec.recommendation_stage
    snapshot["environment_hash"] = environment_hash(settings)
    if getattr(spec, "dialogue_mode", "fixed") == "simulated":
        if not snapshot["persona"]["base_url"] or not snapshot["persona"]["model"]:
            raise ValueError(
                "페르소나용 Gemini 연결이 없습니다. 기존 Gemini 서비스 설정 또는 "
                "PERSONA_MODEL_BASE_URL / PERSONA_MODEL_NAME을 설정하세요."
            )
        if not model_key(snapshot["persona"], settings):
            raise ValueError("페르소나 API 키를 설정하세요.")
    if spec.recommendation_stage == "recommendation" and not snapshot["catalog_configured"]:
        raise ValueError(
            "실제 상품 추천에는 CATALOG_DATABASE_URL_RO 또는 CATALOG_DATABASE_URL이 필요합니다."
        )
    return snapshot


def model_key(config, settings):
    if config.get("credential_source") == "connection":
        return connection_key(config, settings)
    secret = settings.model_api_key or settings.openrouter_api_key
    return secret.get_secret_value() if secret else None
