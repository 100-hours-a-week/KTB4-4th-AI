# Need U AI Server

Spring Backend와 내부 HTTP로 통신하는 FastAPI 서비스다. V1에서는 외부
OpenAI-compatible 모델 API를 호출하고, V2에서는 같은 포트 계약을 유지한 채 모델
endpoint와 model name만 자체 GPU 서버 값으로 교체한다.

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
```

## 패키지 경계

```text
app/api             HTTP, SSE, 인증 의존성, 요청·응답 스키마
app/application     유스케이스 조립과 외부 시스템 포트
app/domain          대화·프로필·추천·카탈로그 순수 규칙
app/infrastructure  Redis·pgvector·모델 API 포트 구현
app/core            환경설정, 공통 오류, 미들웨어
```

의존 방향은 `api -> application -> domain`이다. `infrastructure`는
`application/ports`의 인터페이스를 구현하며, `domain`은 FastAPI·Redis·DB를 import하지
않는다.
