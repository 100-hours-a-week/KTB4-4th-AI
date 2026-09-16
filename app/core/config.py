from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, RedisDsn, SecretStr
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
    ai_database_url: str | None = None

    response_model_base_url: AnyHttpUrl | None = None
    extraction_model_base_url: AnyHttpUrl | None = None
    model_api_key: SecretStr | None = None
    response_model_name: str = "response-model"
    extraction_model_name: str = "extraction-model"


@lru_cache
def get_settings() -> Settings:
    return Settings()
