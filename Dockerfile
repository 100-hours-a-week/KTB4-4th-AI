# syntax=docker/dockerfile:1

# =========================================================
# 1. Build Stage
# =========================================================

FROM python:3.12.11-slim AS builder

WORKDIR /service

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app ./app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


# =========================================================
# 2. Runtime Stage
# =========================================================

FROM python:3.12.11-slim AS runtime

WORKDIR /service

RUN groupadd --system app \
    && useradd --system --gid app app

COPY --from=builder --chown=app:app /service /service

ENV PATH="/service/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]