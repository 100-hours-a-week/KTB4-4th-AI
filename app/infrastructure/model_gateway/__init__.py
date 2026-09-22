"""OpenAI-compatible model adapters; V1 and V2 differ only by settings."""

from app.infrastructure.model_gateway.openai_compatible import (
    ModelGatewayError,
    OpenAICompatibleModelGateway,
)

__all__ = ["ModelGatewayError", "OpenAICompatibleModelGateway"]
