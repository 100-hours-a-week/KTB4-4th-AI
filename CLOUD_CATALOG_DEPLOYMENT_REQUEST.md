# AI 카탈로그 운영 적재를 위한 클라우드 요청 사항

## 실행 구조

- FastAPI는 기존 V1 EC2의 Docker 컨테이너에서 실행합니다. Docker Compose의 `network_mode: host` 구성은 유지합니다.
- 상품 문서는 미리 작성해 둔 파일(JSONL)로 적재합니다. 적재할 때는 임베딩(Upstage)만 호출하고 문서 생성 모델은 쓰지 않습니다.
- 적재는 AI 담당이 SSM 포트 포워딩으로 PostgreSQL에 접속해 `catalog_import` 잡으로 실행합니다.
- 클라우드에는 스키마 변경, 데이터 비우기, 권한, 시크릿, 접속 경로를 요청합니다.

## 1. PostgreSQL 마이그레이션과 권한

적용 이력을 먼저 확인하고, DDL 권한이 있는 인프라 계정으로 아래 마이그레이션 중 미적용분만 순서대로
적용해 주세요. 이미 적용된 파일은 다시 실행하지 않습니다.

1. `migrations/001_create_catalog_products.sql`
2. `migrations/002_add_catalog_sync_state.sql`
3. `migrations/003_create_catalog_product_documents.sql`
4. `migrations/004_add_catalog_categories.sql`
5. `migrations/005_add_catalog_subcategories.sql`

004, 005 적용 여부는 아래 쿼리로 확인합니다. 6줄이면 005까지 적용된 상태입니다.

```sql
SELECT column_name FROM information_schema.columns
 WHERE table_name = 'catalog_products'
   AND column_name IN ('category', 'themes', 'gift_available',
                       'subcategory', 'dedup_group', 'is_generic_voucher');
```

새 API는 004에서 추가한 `category`, `gift_available`을 읽습니다. 004를 적용하기 전에 배포하면 추천 검색이
실패합니다.

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
- 애플리케이션 계정에는 `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, `CREATE EXTENSION` 권한을 주지 않습니다.

## 2. 상품 전체 재적재

상품을 새로 채울 때만 진행합니다.

1. AI PostgreSQL 스냅샷을 남깁니다.
2. 인프라 계정으로 카탈로그 데이터를 비웁니다. 세 테이블이 FK로 엮여 있어 한 문장으로 비웁니다.
   `catalog_embedding_models`는 문서 테이블이 참조하므로 비우지 않습니다.

```sql
TRUNCATE catalog_product_documents, catalog_products, catalog_sync_runs;
```

비운 뒤 적재가 끝날 때까지 추천 결과가 비어 있습니다. 상품 약 8천 개 기준 수 분이 걸립니다.

## 3. SSM Parameter Store

아래 값을 `/needu/prod/ai/*` 경로에 둡니다. 비밀값은 `SecureString`으로 저장합니다.

```text
/needu/prod/ai/UPSTAGE_API_KEY=<SecureString>
/needu/prod/ai/EMBEDDING_BASE_URL=https://api.upstage.ai/v1
/needu/prod/ai/EMBEDDING_SPACE_ID=solar-embedding-2
/needu/prod/ai/EMBEDDING_PASSAGE_MODEL_NAME=solar-embedding-2-passage
/needu/prod/ai/EMBEDDING_QUERY_MODEL_NAME=solar-embedding-2-query
/needu/prod/ai/EMBEDDING_DIMENSIONS=1024
/needu/prod/ai/SOURCE_MYSQL_CATEGORY_COLUMN=category
/needu/prod/ai/SOURCE_MYSQL_THEMES_COLUMN=gift_themes
```

`UPSTAGE_API_KEY`와 `EMBEDDING_*`는 추천 API가 검색 문장을 임베딩할 때도 씁니다.
`SOURCE_MYSQL_*_COLUMN`은 정기 동기화가 원본의 카테고리와 선물 테마를 읽을 컬럼입니다.

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

추천 API는 `CATALOG_DATABASE_URL_RO`, 동기화·적재 배치는 `CATALOG_DATABASE_URL`을 사용합니다.

## 4. SSM 포트 포워딩

AI 담당이 로컬에서 PostgreSQL 5432 포트로 포워딩 세션을 열 수 있어야 합니다.

요청 사항:

- AI 담당 IAM 사용자에 대상 인스턴스로 `ssm:StartSession` 허용
  (`AWS-StartPortForwardingSession` 또는 `AWS-StartPortForwardingSessionToRemoteHost` 문서)
- 대상 인스턴스 ID 공유
- `ai_catalog_batch` 접속 정보는 메신저가 아닌 방법으로 전달
- 5432 포트의 외부 Security Group 인바운드는 열지 않음

```bash
aws ssm start-session \
  --target <INSTANCE_ID> \
  --document-name AWS-StartPortForwardingSession \
  --parameters portNumber=5432,localPortNumber=15432
```

## 5. 적재 실행과 검증

AI 담당이 포워딩된 포트로 접속해 실행합니다.

```bash
CATALOG_DATABASE_URL=postgresql://ai_catalog_batch:<password>@127.0.0.1:15432/needu_vector \
  uv run python -m app.jobs.catalog_import \
  --products <상품 CSV> --documents <문서 JSONL>
```

검증 기준:

- `catalog_products.processing_status = 'ready'`가 상품 수와 같음
- 활성 상품마다 `content`, `usage` 문서가 1개, `gift_available` 상품은 `gift` 문서도 1개
- `vector_dims(embedding) = 1024`
- API 컨테이너가 `CATALOG_DATABASE_URL_RO` 계정으로 추천 조회 가능

```sql
SELECT processing_status, count(*) FROM catalog_products GROUP BY 1;
SELECT space, count(*), min(vector_dims(embedding)), max(vector_dims(embedding))
  FROM catalog_product_documents GROUP BY 1;
```

## 6. 정기 실행 관련

원본 상품이 바뀔 때 반영하는 정기 실행 순서는 `catalog_sync` 완료 후 `catalog_enrich`입니다.
`catalog_enrich`는 문서 생성 모델 엔드포인트(`DOCUMENT_MODEL_*`)가 필요합니다. 항상 접근 가능한 운영용
문서 생성 모델이 준비된 뒤 자동 스케줄링을 구성합니다.
