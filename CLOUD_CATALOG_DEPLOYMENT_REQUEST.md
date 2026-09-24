# AI 카탈로그 운영 적재를 위한 클라우드 요청 사항

## 실행 구조

- FastAPI와 카탈로그 배치는 기존 V1 EC2의 Docker 컨테이너에서 실행합니다.
- 운영 MySQL 조회, Upstage 임베딩 호출, PostgreSQL pgvector 적재는 EC2 컨테이너에서 수행합니다.
- 설명 생성 모델 `google/gemma-4-12b-qat`만 개발자 Mac의 LM Studio에서 실행합니다.
- Mac LM Studio는 인터넷에 공개하지 않고 SSH reverse tunnel을 통해 EC2의 `127.0.0.1:1234`에
  연결합니다.
- Docker Compose의 `network_mode: host` 구성은 유지합니다.

## 1. PostgreSQL 마이그레이션과 권한

적용 이력을 먼저 확인하고, DDL 권한이 있는 인프라 계정으로 아래 마이그레이션 중 미적용분만 순서대로
적용해 주세요. 이미 적용된 파일은 다시 실행하지 않습니다.

1. `migrations/001_create_catalog_products.sql`
2. `migrations/002_add_catalog_sync_state.sql`
3. `migrations/003_create_catalog_product_documents.sql`

확인 사항:

- `vector` extension 0.8.1 사용
- `ai_catalog_batch`
  - `catalog_products`: `SELECT`, `INSERT`, `UPDATE`
  - `catalog_sync_runs`: `SELECT`, `INSERT`, `UPDATE`
  - `catalog_product_documents`: `SELECT`, `INSERT`, `UPDATE`
  - `catalog_embedding_models`: `SELECT`
  - `catalog_sync_runs_run_id_seq`: `USAGE`, `SELECT`
- `ai_catalog_api`
  - 위 카탈로그 테이블에 `SELECT`
- 애플리케이션 계정에는 `CREATE`, `ALTER`, `DROP`, `CREATE EXTENSION` 권한을 주지 않습니다.

## 2. SSM Parameter Store 추가 값

아래 값을 `/needu/prod/ai/*` 경로에 추가해 주세요. 비밀값은 `SecureString`으로 저장합니다.

```text
/needu/prod/ai/DOCUMENT_MODEL_BASE_URL=http://127.0.0.1:1234/v1
/needu/prod/ai/DOCUMENT_MODEL_NAME=google/gemma-4-12b-qat
/needu/prod/ai/DOCUMENT_MODEL_MAX_TOKENS=2048
/needu/prod/ai/DOCUMENT_MODEL_CONCURRENCY=1
/needu/prod/ai/DOCUMENT_MODEL_ENABLE_THINKING=false
/needu/prod/ai/DOCUMENT_MODEL_TIMEOUT_SECONDS=180
/needu/prod/ai/CATALOG_ENRICH_BATCH_SIZE=30
/needu/prod/ai/CATALOG_ENRICH_STALE_SECONDS=3600
/needu/prod/ai/UPSTAGE_API_KEY=<SecureString>
```

현재 기본값과 운영값을 명시적으로 맞추기 위해 다음 값도 권장합니다.

```text
/needu/prod/ai/EMBEDDING_BASE_URL=https://api.upstage.ai/v1
/needu/prod/ai/EMBEDDING_SPACE_ID=solar-embedding-2
/needu/prod/ai/EMBEDDING_PASSAGE_MODEL_NAME=solar-embedding-2-passage
/needu/prod/ai/EMBEDDING_QUERY_MODEL_NAME=solar-embedding-2-query
/needu/prod/ai/EMBEDDING_DIMENSIONS=1024
```

기존에 등록된 다음 값은 그대로 사용합니다.

```text
CATALOG_DATABASE_URL
CATALOG_DATABASE_URL_RO
SOURCE_MYSQL_HOST
SOURCE_MYSQL_PORT
SOURCE_MYSQL_DATABASE
SOURCE_MYSQL_TABLE
SOURCE_MYSQL_USER
SOURCE_MYSQL_PASSWORD
```

## 3. Docker Compose 환경변수 전달

SSM에서 읽은 아래 값을 AI 컨테이너에 전달해 주세요.

```text
DOCUMENT_MODEL_BASE_URL
DOCUMENT_MODEL_NAME
DOCUMENT_MODEL_MAX_TOKENS
DOCUMENT_MODEL_CONCURRENCY
DOCUMENT_MODEL_ENABLE_THINKING
DOCUMENT_MODEL_TIMEOUT_SECONDS
CATALOG_ENRICH_BATCH_SIZE
CATALOG_ENRICH_STALE_SECONDS
UPSTAGE_API_KEY
EMBEDDING_BASE_URL
EMBEDDING_SPACE_ID
EMBEDDING_PASSAGE_MODEL_NAME
EMBEDDING_QUERY_MODEL_NAME
EMBEDDING_DIMENSIONS
```

추천 API는 `CATALOG_DATABASE_URL_RO`, 동기화·적재 배치는 `CATALOG_DATABASE_URL`을 사용합니다.

## 4. SSH reverse tunnel 허용

초기 적재 작업 동안 개발자 Mac에서 EC2로 다음 reverse tunnel을 열 수 있어야 합니다.

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:1234:127.0.0.1:1234 \
  <EC2_USER>@<EC2_HOST>
```

요청 사항:

- SSH 서버의 loopback reverse forwarding 허용
- 필요 시 `AllowTcpForwarding yes` 확인
- 원격 포트는 `127.0.0.1:1234`에만 바인딩
- 1234 포트의 외부 Security Group 인바운드는 열지 않음
- 개발자 SSH 접근은 승인된 IP 또는 현재 팀의 기존 접속 방식으로 제한

터널 연결 후 EC2 호스트와 컨테이너에서 아래 요청이 성공해야 합니다.

```bash
curl http://127.0.0.1:1234/v1/models
```

## 5. 최초 운영 실행 순서

터널 연결과 LM Studio 모델 로드를 확인한 뒤 EC2 컨테이너에서 실행합니다.

```bash
python -m app.jobs.catalog_sync --mode full
python -m app.jobs.catalog_enrich --max-batches 999
```

실패 건 재시도:

```bash
python -m app.jobs.catalog_enrich --max-batches 999 --retry-failed
```

최초 검증 기준:

- `catalog_products.processing_status = 'failed'`가 0건
- 활성 상품마다 `content`, `usage`, `gift` 문서가 각각 1개
- `vector_dims(embedding) = 1024`
- API 컨테이너가 `CATALOG_DATABASE_URL_RO` 계정으로 추천 조회 가능

로컬 테스트용 `seed_products.sql` 데이터와 로컬 PostgreSQL 데이터는 운영 DB로 복사하지 않습니다.
운영 MySQL의 실제 상품을 `catalog_sync`로 읽어 운영 카탈로그를 생성합니다.

## 6. 정기 실행 관련

LM Studio가 개발자 Mac에 있는 동안에는 터널이 연결된 시간에만 수동 배치를 실행합니다. 자동
스케줄링은 항상 접근 가능한 운영용 모델 엔드포인트가 준비된 뒤 구성합니다. 정기 실행 순서는 항상
`catalog_sync` 완료 후 `catalog_enrich`입니다.
