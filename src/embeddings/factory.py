"""Pick the embedding provider from EMBEDDING_PROVIDER (default: local)."""

from __future__ import annotations

import os

from src.monitoring.logger import StructuredLogger

logger = StructuredLogger(__name__)


def get_embedding_provider(provider: str | None = None):
    """
    Return the embedding provider named by `provider` or EMBEDDING_PROVIDER.

    'local' (default) runs sentence-transformers on this machine and needs no
    keys. 'bedrock' calls Amazon Titan Embeddings V2 and needs AWS credentials.
    """
    active = provider or os.getenv("EMBEDDING_PROVIDER", "local")

    if active == "bedrock":
        from src.embeddings.bedrock import BedrockEmbeddingProvider

        logger.info("Embedding provider: AWS Bedrock (Titan V2)")
        return BedrockEmbeddingProvider()
    if active == "local":
        from src.embeddings.local import LocalEmbeddingProvider

        logger.info("Embedding provider: local sentence-transformers")
        return LocalEmbeddingProvider()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {active!r}. Valid values: 'local' (default), 'bedrock'."
    )
