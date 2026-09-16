from typing import Literal

from app.api.schemas.common import ApiModel


class HealthResponse(ApiModel):
    status: Literal["ok"]
    version: str
