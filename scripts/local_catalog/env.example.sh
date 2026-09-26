# 로컬 카탈로그 적재 테스트용 환경 변수 예시.
#   cp scripts/local_catalog/env.example.sh scripts/local_catalog/env.sh
#   source scripts/local_catalog/env.sh

export LOCAL_CATALOG_ROOT="${LOCAL_CATALOG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/../.." && pwd)}"

# 원본: 테스트 전용 MySQL 인스턴스 (포트 3307, 읽기 전용 계정)
export SOURCE_MYSQL_HOST=127.0.0.1
export SOURCE_MYSQL_PORT=3307
export SOURCE_MYSQL_USER=needu_reader
export SOURCE_MYSQL_PASSWORD="${SOURCE_MYSQL_PASSWORD:-}"
export SOURCE_MYSQL_DATABASE=needu_source
export SOURCE_MYSQL_TABLE=products

# 적재 대상: 로컬 PostgreSQL + pgvector
export CATALOG_DATABASE_URL="postgresql://$(whoami)@127.0.0.1:5432/needu_catalog_local"
export CATALOG_ENRICH_BATCH_SIZE=30

# LM Studio 로컬 서버
export DOCUMENT_MODEL_BASE_URL="${DOCUMENT_MODEL_BASE_URL:-http://127.0.0.1:1234/v1}"
export DOCUMENT_MODEL_NAME="${DOCUMENT_MODEL_NAME:-kanana-1.5-8b-instruct-2505}"
export DOCUMENT_MODEL_MAX_TOKENS="${DOCUMENT_MODEL_MAX_TOKENS:-1024}"
export DOCUMENT_MODEL_CONCURRENCY="${DOCUMENT_MODEL_CONCURRENCY:-1}"
export DOCUMENT_MODEL_ENABLE_THINKING="${DOCUMENT_MODEL_ENABLE_THINKING:-false}"
export DOCUMENT_MODEL_TEMPERATURE="${DOCUMENT_MODEL_TEMPERATURE:-0.2}"
export DOCUMENT_MODEL_STOP_SEQUENCE="${DOCUMENT_MODEL_STOP_SEQUENCE:-<|eot_id|>}"
export DOCUMENT_MODEL_TIMEOUT_SECONDS="${DOCUMENT_MODEL_TIMEOUT_SECONDS:-180}"
