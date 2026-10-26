"""
Local embedding provider using sentence-transformers.

This is the default provider. Everything runs offline, no AWS account,
no API keys, no internet required after the model downloads on first run.

The model (all-MiniLM-L6-v2) produces 384-dimensional vectors. Smaller
than Bedrock Titan (1024d) but plenty for development and fast enough to
iterate on chunking strategies without burning cloud credits.

First run: downloads ~90MB model to ~/.cache/torch/sentence_transformers.
Subsequent runs: loads from disk cache in ~1s.

To use Bedrock instead:
    EMBEDDING_PROVIDER=bedrock  # see src/embeddings/bedrock.py
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from src.monitoring.logger import StructuredLogger

logger = StructuredLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class LocalEmbeddingProvider:
    """
    Wraps sentence-transformers with the same interface as BedrockEmbeddingProvider
    so the storage layer doesn't care which backend is active.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or os.getenv("LOCAL_EMBED_MODEL", DEFAULT_MODEL)
        self._model = None  # lazy load, don't pay startup cost if unused

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading local embedding model", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("Model loaded", model=self._model_name, dim=EMBEDDING_DIM)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_model()
        assert self._model is not None

        embeddings = self._model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        result = self.embed_texts([text])
        return result[0]

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM
