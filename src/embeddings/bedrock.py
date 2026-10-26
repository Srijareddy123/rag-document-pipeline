"""
AWS Bedrock embedding provider (optional).

Demo mode never touches this module. The factory in src/embeddings/factory.py
returns LocalEmbeddingProvider unless EMBEDDING_PROVIDER=bedrock.

To switch Bedrock on:
    export EMBEDDING_PROVIDER=bedrock
    export AWS_REGION=us-east-1
    # AWS credentials must be configured via IAM role, instance profile,
    # or ~/.aws/credentials, never hardcoded.

Why Bedrock over OpenAI:
    - Stays inside the AWS account, same IAM, same VPC, no new vendor
    - Titan Embeddings V2 was competitive on MTEB in late 2024
    - On-demand pricing fit batch workloads better than per-token OpenAI costs
    - Data residency stays within the AWS region

The interface is identical to LocalEmbeddingProvider so the pipeline
doesn't know or care which backend is active.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

from src.ingestion.retry import RetryableIngestionError, with_retry
from src.monitoring.logger import StructuredLogger

logger = StructuredLogger(__name__)

BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
BEDROCK_EMBED_DIM = 1024


class BedrockEmbeddingProvider:
    """
    Embedding provider using Amazon Titan Embeddings V2.

    Optional. Requires AWS credentials and network access to Bedrock.
    All calls go through with_retry() to handle ThrottlingException and
    transient ServiceUnavailableException.
    """

    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
    ):
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._model_id = model_id or os.getenv("BEDROCK_EMBED_MODEL", BEDROCK_EMBED_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 not installed. Run: pip install boto3"
            ) from exc
        self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    @with_retry(max_attempts=3, base_delay=2.0, operation_name="bedrock_embed")
    def _embed_single(self, text: str) -> list[float]:
        client = self._get_client()
        body = json.dumps({"inputText": text})
        try:
            response = client.invoke_model(
                modelId=self._model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except Exception as exc:
            err_str = str(exc)
            if any(kw in err_str for kw in ("ThrottlingException", "ServiceUnavailable", "Too Many Requests")):
                raise RetryableIngestionError(f"Bedrock throttled: {exc}") from exc
            raise

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = []
        for i, text in enumerate(texts):
            if not text.strip():
                # Bedrock rejects empty strings, substitute zero vector
                logger.warning("Empty text at index, using zero vector", index=i)
                embeddings.append([0.0] * BEDROCK_EMBED_DIM)
                continue
            embeddings.append(self._embed_single(text))
        logger.info("Bedrock embedding batch complete", count=len(embeddings))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        return BEDROCK_EMBED_DIM
