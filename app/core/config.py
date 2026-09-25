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

    # 백엔드 MySQL 원본. 같은 EC2에 있으면 127.0.0.1 로 붙는다.
    # 읽기 전용 계정을 쓴다. 이 경로로는 절대 쓰기를 하지 않는다.
    source_mysql_host: str | None = None
    source_mysql_port: int = Field(default=3306, gt=0, le=65535)
    source_mysql_user: str | None = None
    source_mysql_password: SecretStr | None = None
    source_mysql_database: str | None = None
    source_mysql_table: str = "products"
    source_mysql_pool_size: int = Field(default=2, ge=1, le=20)

    # AI 카탈로그 PostgreSQL. 예: postgresql://user:pw@host:5432/ai_catalog
    catalog_database_url: str | None = None
    catalog_database_url_ro: str | None = None
    catalog_pool_min_size: int = Field(default=1, ge=1)
    catalog_pool_max_size: int = Field(default=5, ge=1)
    catalog_sync_batch_size: int = Field(default=1000, ge=1, le=10000)
    # 문서 생성 배치가 한 번에 집어가는 상품 수. LLM 속도에 맞춰 조절한다.
    catalog_enrich_batch_size: int = Field(default=50, ge=1, le=500)
    catalog_enrich_stale_seconds: int = Field(default=3600, ge=60)

    response_model_base_url: AnyHttpUrl | None = None
    extraction_model_base_url: AnyHttpUrl | None = None
    model_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    response_model_name: str = "response-model"
    extraction_model_name: str = "extraction-model"

    response_model_timeout_seconds: float = Field(default=60.0, gt=0)
    extraction_model_timeout_seconds: float = Field(default=30.0, gt=0)
    summary_model_timeout_seconds: float = Field(default=5.0, gt=0)
    # 카탈로그 문서 생성용 로컬 LLM. LM Studio 기본값은 http://127.0.0.1:1234/v1 이다.
    document_model_base_url: AnyHttpUrl | None = None
    document_model_name: str = "document-model"
    document_model_max_tokens: int = Field(default=1024, gt=0)
    document_model_concurrency: int = Field(default=4, ge=1)
    document_model_enable_thinking: bool | None = None
    document_model_timeout_seconds: float = Field(default=180.0, gt=0)

    response_model_max_tokens: int = Field(default=2048, gt=0)
    extraction_model_max_tokens: int = Field(default=2048, gt=0)

    embedding_base_url: AnyHttpUrl = AnyHttpUrl("https://api.upstage.ai/v1")
    # 짝으로 학습된 두 모델이 같은 벡터 공간을 쓴다. space id는 DB의 embedding_model_version 값이다.
    embedding_space_id: str = "solar-embedding-2"
    embedding_passage_model_name: str = "solar-embedding-2-passage"
    embedding_query_model_name: str = "solar-embedding-2-query"
    embedding_dimensions: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=100, ge=1, le=100)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    upstage_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
