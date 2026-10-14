"""Resolve connection credentials without storing keys in the endpoint registry."""

import os

from admin_api.store import ROOT
from app.core.config import Settings


def connection_key(connection, settings=None):
    name = connection.get("api_key_env")
    if not name:
        return None
    key = os.environ.get(name)
    if not key and name in {"HF_TOKEN", "OPENROUTER_API_KEY", "AI_LAB_COLAB_KEY"}:
        configured = settings or Settings(_env_file=ROOT / ".env")
        secret = getattr(configured, name.lower())
        key = secret.get_secret_value() if secret else None
    if not key:
        raise ValueError(f"API key 환경변수 {name}가 설정되지 않았습니다.")
    return key
