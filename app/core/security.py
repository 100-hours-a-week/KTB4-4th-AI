from hmac import compare_digest

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.errors import ApiError

bearer_scheme = HTTPBearer(auto_error=False)


async def require_service_token(request: Request) -> None:
    credentials: HTTPAuthorizationCredentials | None = await bearer_scheme(request)
    configured_token = request.app.state.settings.service_token

    if configured_token is None or credentials is None:
        raise ApiError(
            status_code=401,
            code="UNAUTHORIZED",
            message="인증 정보가 올바르지 않습니다.",
        )

    if credentials.scheme.lower() != "bearer" or not compare_digest(
        credentials.credentials, configured_token.get_secret_value()
    ):
        raise ApiError(
            status_code=401,
            code="UNAUTHORIZED",
            message="인증 정보가 올바르지 않습니다.",
        )
