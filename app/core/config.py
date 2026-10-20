from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, RedisDsn, SecretStr, model_validator
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
    # 관심사별 검색 의도를 추출 모델로 만들어 Redis에 캐시한다. 끄면 추천은 템플릿 쿼리만 쓴다.
    search_intent_enabled: bool = True
    ai_database_url: str | None = None

    # 백엔드 MySQL 원본. 같은 EC2에 있으면 127.0.0.1 로 붙는다.
    # 읽기 전용 계정을 쓴다. 이 경로로는 절대 쓰기를 하지 않는다.
    source_mysql_host: str | None = None
    source_mysql_port: int = Field(default=3306, gt=0, le=65535)
    source_mysql_user: str | None = None
    source_mysql_password: SecretStr | None = None
    source_mysql_database: str | None = None
    source_mysql_table: str = "products"
    # 상품 카테고리 컬럼. 비우면 카테고리 없이 읽고 문서 생성 모델이 고른다.
    source_mysql_category_column: str | None = None
    # 선물 테마 목록 컬럼(JSON 배열 또는 | 구분). gift 문서 작성 여부와 힌트에 쓴다.
    source_mysql_themes_column: str | None = None
    source_mysql_pool_size: int = Field(default=2, ge=1, le=20)

    # AI 카탈로그 PostgreSQL. 예: postgresql://user:pw@host:5432/ai_catalog
    catalog_database_url: str | None = None
    catalog_database_url_ro: str | None = None
    catalog_pool_min_size: int = Field(default=1, ge=1)
    catalog_pool_max_size: int = Field(default=5, ge=1)
    catalog_sync_batch_size: int = Field(default=1000, ge=1, le=10000)
    # 문서 생성 배치가 한 번에 집어가는 상품 수. LLM 속도에 맞춰 조절한다.
    catalog_enrich_batch_size: int = Field(default=50, ge=1, le=500)
    catalog_enrich_embedding_concurrency: int = Field(default=4, ge=1, le=20)
    catalog_enrich_progress_interval: int = Field(default=100, ge=1)
    catalog_enrich_stale_seconds: int = Field(default=3600, ge=60)

    response_model_base_url: AnyHttpUrl | None = None
    extraction_model_base_url: AnyHttpUrl | None = None
    model_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    ai_lab_colab_key: SecretStr | None = None
    response_model_name: str = "response-model"
    extraction_model_name: str = "extraction-model"

    # 응답·추출 1차 엔드포인트(RunPod vLLM)가 꺼졌거나 실패하면 넘길 엔드포인트(OpenRouter).
    # 주소와 모델명을 함께 넣어야 켜지고, 키는 OPENROUTER_API_KEY를 쓴다. 비우면 1차만 쓴다.
    response_fallback_model_base_url: AnyHttpUrl | None = None
    response_fallback_model_name: str | None = None
    extraction_fallback_model_base_url: AnyHttpUrl | None = None
    extraction_fallback_model_name: str | None = None
    # 폴백이 있을 때만 1차 호출에 거는 제한. read는 응답 바이트 사이의 대기 한도다.
    primary_model_connect_timeout_seconds: float = Field(default=3.0, gt=0)
    primary_model_read_timeout_seconds: float = Field(default=20.0, gt=0)
    # 1차가 실패한 뒤 다시 시도하기 전까지 폴백으로만 보내는 시간. 실패가 이어지면 두 배씩
    # 최대값까지 늘린다. 최대값은 GPU를 켠 뒤 1차로 돌아오기까지 걸리는 최대 지연이기도 하다.
    primary_model_cooldown_seconds: float = Field(default=30.0, ge=0)
    primary_model_max_cooldown_seconds: float = Field(default=120.0, ge=0)
    # RunPod Pod 조회. 키와 이름 접두어를 함께 넣으면 응답·추출 1차 주소를 위의 BASE_URL 대신
    # 이름이 접두어로 시작하는 RUNNING Pod에서 찾는다. Pod ID가 바뀌어도 재배포가 없다.
    # 키는 RunPod 콘솔의 Restricted 키(Pod 읽기 전용)를 쓴다.
    runpod_api_key: SecretStr | None = None
    runpod_pod_name_prefix: str | None = None
    runpod_pod_port: int = Field(default=8000, gt=0, le=65535)
    runpod_api_base_url: AnyHttpUrl = AnyHttpUrl("https://api.runpod.io")
    runpod_lookup_interval_seconds: float = Field(default=30.0, gt=0)

    # 취향 판단 모델(TypeSafe Jev). OpenRouter의 /systemone 으로 호출하고 키는 위와 같은 것을 쓴다.
    # base url이 없으면 판단 단계 없이 추출 모델 혼자 취향을 뽑는 기존 방식으로 돈다.
    judgment_model_base_url: AnyHttpUrl | None = None
    judgment_model_name: str = "~typesafe/jev-latest"

    response_model_timeout_seconds: float = Field(default=60.0, gt=0)
    extraction_model_timeout_seconds: float = Field(default=30.0, gt=0)
    judgment_model_timeout_seconds: float = Field(default=10.0, gt=0)
    # 취향 요약이 3~4문장으로 길어져서 5초로는 잘릴 수 있다.
    summary_model_timeout_seconds: float = Field(default=10.0, gt=0)
    # AI Lab의 사용자 시뮬레이터. 명시하면 응답·추출 모델과 독립적으로 유지한다.
    persona_model_base_url: AnyHttpUrl | None = None
    persona_model_name: str | None = None
    persona_model_api_key_env: str = Field(default="", pattern=r"^$|^[A-Za-z_][A-Za-z0-9_]*$")
    # 카탈로그 문서 생성용 로컬 LLM. LM Studio 기본값은 http://127.0.0.1:1234/v1 이다.
    document_model_base_url: AnyHttpUrl | None = None
    document_model_name: str = "document-model"
    document_model_max_tokens: int = Field(default=1024, gt=0)
    document_model_concurrency: int = Field(default=4, ge=1)
    document_model_enable_thinking: bool | None = None
    document_model_temperature: float | None = Field(default=None, ge=0, le=2)
    document_model_stop_sequence: str | None = None
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

    @model_validator(mode="after")
    def _fallback_needs_url_and_name(self) -> "Settings":
        for role in ("response", "extraction"):
            url = getattr(self, f"{role}_fallback_model_base_url")
            name = getattr(self, f"{role}_fallback_model_name")
            if (url is None) != (not name):
                raise ValueError(
                    f"{role.upper()}_FALLBACK_MODEL_BASE_URL and "
                    f"{role.upper()}_FALLBACK_MODEL_NAME must be set together"
                )
        if self.runpod_pod_name_prefix and self.runpod_api_key is None:
            raise ValueError("RUNPOD_POD_NAME_PREFIX needs RUNPOD_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
