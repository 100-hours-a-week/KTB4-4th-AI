# Need U AI Server

Spring Backend와 내부 HTTP로 통신하는 FastAPI 서비스다. V1에서는 외부 OpenAI-compatible 모델 API를
호출하고, V2에서는 같은 포트 계약을 유지한 채 모델 endpoint와 model name만 자체 GPU 서버 값으로
교체한다.

## 로컬 실행

Python 3.12와 `uv`를 기준으로 한다.

```bash
cp .env.example .env
uv sync --dev
uv run uvicorn app.main:app --reload
```

헬스 체크:

```bash
curl http://localhost:8000/health
```

테스트와 정적 검사:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
npm ci
npm run lint:js
npm run format:check
```

Python 코드는 Ruff가 검사·포맷한다. ESLint는 Semantic Release JavaScript 설정을 검사하고, Prettier는
설정 파일과 저장소의 JSON·YAML·Markdown 파일을 검사한다.

## 커밋과 릴리스

`main` 브랜치의 Semantic Release는 Conventional Commit을 기준으로 Git 태그와 GitHub Release를
생성한다. 이 저장소는 npm 또는 PyPI 패키지를 배포하지 않는다.

```text
fix: 오류 수정                 → patch
feat: 기능 추가                → minor
feat!: 호환되지 않는 계약 변경 → major
```

## 패키지 경계

```text
app/api             HTTP, SSE, 인증 의존성, 요청·응답 스키마
app/application     유스케이스 조립과 외부 시스템 포트
app/domain          대화·프로필·추천·카탈로그 순수 규칙
app/infrastructure  Redis·pgvector·모델 API 포트 구현
app/core            환경설정, 공통 오류, 미들웨어
```

의존 방향은 `api -> application -> domain`이다. `infrastructure`는 `application/ports`의
인터페이스를 구현하며, `domain`은 FastAPI·Redis·DB를 import하지 않는다.

## 운영 카탈로그 적재

운영 적재 작업은 EC2 컨테이너에서 실행한다. MySQL과 PostgreSQL 자격 증명은 EC2에만 두고, 설명 생성용
LM Studio만 개발자 Mac에서 실행한다. Mac의 LM Studio는 외부에 공개하지 않고 SSH reverse tunnel로
EC2의 loopback 포트에 연결한다.

Mac에서 LM Studio 서버와 `google/gemma-4-12b-qat` 모델을 실행한 다음 터널을 연다.

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:1234:127.0.0.1:1234 \
  <EC2_USER>@<EC2_HOST>
```

EC2 호스트와 host network 컨테이너에서는 `http://127.0.0.1:1234/v1`로 LM Studio에 접근한다. 먼저
EC2에서 연결을 확인한다.

```bash
curl http://127.0.0.1:1234/v1/models
```

마이그레이션은 DDL 권한이 있는 인프라 계정으로 먼저 적용한다. 이후 EC2 컨테이너에서 운영 MySQL
전체를 PostgreSQL 카탈로그에 동기화하고, `pending` 상품만 설명 생성과 임베딩을 수행한다.

```bash
python -m app.jobs.catalog_sync --mode full
python -m app.jobs.catalog_enrich --max-batches 999
```

터널 또는 외부 API 문제로 실패한 상품만 다시 처리할 때는 다음 옵션을 쓴다.

```bash
python -m app.jobs.catalog_enrich --max-batches 999 --retry-failed
```

전체 동기화는 모든 MySQL 행을 읽지만 변경되지 않은 상품은 LLM과 임베딩을 다시 호출하지 않는다.
상품명 또는 가격 구간이 달라진 상품만 다시 처리한다. 임베딩 모델을 바꾸는 경우에는 기존 추천을
유지하면서 새 모델 버전을 채우고 전환해야 하므로 별도 전체 백필 작업으로 진행한다.
