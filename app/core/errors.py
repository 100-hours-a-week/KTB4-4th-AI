from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas.common import ErrorResponse


class ApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "req_unknown")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        body = ErrorResponse(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            request_id=_request_id(request),
            details=error.details,
        )
        return JSONResponse(status_code=error.status_code, content=body.model_dump(by_alias=True))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(
            code="INVALID_REQUEST",
            message="요청 형식이 올바르지 않습니다.",
            retryable=False,
            request_id=_request_id(request),
            details={"errors": error.errors()},
        )
        return JSONResponse(status_code=400, content=body.model_dump(by_alias=True))
