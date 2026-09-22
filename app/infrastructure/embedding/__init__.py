"""OpenAI-compatible embedding adapters; Upstage Solar Embedding 2 by default."""

from app.infrastructure.embedding.openai_compatible import (
    EmbeddingError,
    OpenAICompatibleEmbedder,
)

__all__ = ["EmbeddingError", "OpenAICompatibleEmbedder"]
