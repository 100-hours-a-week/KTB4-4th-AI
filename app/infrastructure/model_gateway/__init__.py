"""OpenAI-compatible model adapters; V1 and V2 differ only by settings."""

from app.infrastructure.model_gateway.failover import FailoverModelGateway, FallbackRoute
from app.infrastructure.model_gateway.openai_compatible import (
    ModelGatewayError,
    OpenAICompatibleModelGateway,
)
from app.infrastructure.model_gateway.runpod_locator import RunPodPodLocator
from app.infrastructure.model_gateway.systemone import SystemOneJudgmentGateway

__all__ = [
    "FailoverModelGateway",
    "FallbackRoute",
    "ModelGatewayError",
    "OpenAICompatibleModelGateway",
    "RunPodPodLocator",
    "SystemOneJudgmentGateway",
]
