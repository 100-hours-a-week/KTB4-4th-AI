from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

JAVA_LONG_MAX = 9_223_372_036_854_775_807
JavaLongId = Annotated[int, Field(strict=True, ge=1, le=JAVA_LONG_MAX)]
UserId = JavaLongId
ConversationRoomId = JavaLongId


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(word.capitalize() for word in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ErrorResponse(ApiModel):
    code: str
    message: str
    retryable: bool
    request_id: str
    details: dict[str, Any] | None = None
