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
