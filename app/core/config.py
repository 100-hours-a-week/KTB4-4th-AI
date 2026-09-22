from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Need U AI Server"
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: str = "INFO"

    service_token: SecretStr | None = None

    redis_url: RedisDsn | None = None
    chat_session_idle_ttl_seconds: int = Field(default=1800, ge=60)
    chat_session_max_lifetime_seconds: int = Field(default=7200, ge=60)
    ai_database_url: str | None = None

    response_model_base_url: AnyHttpUrl | None = None
    extraction_model_base_url: AnyHttpUrl | None = None
    model_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    response_model_name: str = "response-model"
    extraction_model_name: str = "extraction-model"

    response_model_timeout_seconds: float = Field(default=60.0, gt=0)
    extraction_model_timeout_seconds: float = Field(default=30.0, gt=0)
    summary_model_timeout_seconds: float = Field(default=5.0, gt=0)
    response_model_max_tokens: int = Field(default=2048, gt=0)
    extraction_model_max_tokens: int = Field(default=2048, gt=0)

@lru_cache
def get_settings() -> Settings:
    return Settings()
