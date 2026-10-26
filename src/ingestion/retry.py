"""
Retry logic for embedding and storage calls.

Bedrock and ChromaDB both have transient failure modes:
  - Bedrock: ThrottlingException on burst traffic, ServiceUnavailableException
  - ChromaDB: occasional connection resets if the container restarts mid-batch

Exponential backoff with jitter prevents thundering-herd on retries.
Max 3 attempts is intentional, failing fast is better than queuing up
retries that block the rest of the batch for minutes.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

from src.monitoring.logger import StructuredLogger

logger = StructuredLogger(__name__)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Sequence[type[Exception]] = (Exception,),
    operation_name: str = "",
):
    """
    Decorator for exponential backoff with full jitter.

    Full jitter (random(0, delay)) avoids synchronized retries from
    concurrent workers hitting the same rate limit at the same time.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = operation_name or func.__name__
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"{name} failed after {max_attempts} attempts",
                            error=str(exc),
                            attempt=attempt,
                        )
                        raise

                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    jitter = random.uniform(0, delay)
                    logger.warning(
                        f"{name} attempt {attempt} failed, retrying",
                        error=str(exc),
                        retry_in_seconds=round(jitter, 2),
                    )
                    time.sleep(jitter)

            # Should never reach here, but keeps the type checker happy
            raise last_exc  # type: ignore[misc]

        return wrapper
    return decorator


class RetryableIngestionError(Exception):
    """Raised when a document ingestion step fails in a retryable way."""


class PermanentIngestionError(Exception):
    """Raised for non-retryable failures (bad document format, auth errors, etc.)."""
