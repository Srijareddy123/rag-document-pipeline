import json
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class StructuredLogger:
    """JSON-structured logger."""

    def __init__(self, name: str):
        self._logger = get_logger(name)
        self._name = name

    def _emit(self, level: str, message: str, **kwargs: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "logger": self._name,
            "message": message,
            **kwargs,
        }
        getattr(self._logger, level.lower())(json.dumps(record))

    def info(self, message: str, **kwargs: Any) -> None:
        self._emit("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._emit("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._emit("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._emit("DEBUG", message, **kwargs)


def timed(logger: StructuredLogger | None = None):
    """Decorator that logs execution time for ingestion-heavy operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                if logger:
                    logger.info(
                        f"{func.__name__} completed",
                        duration_ms=round(elapsed * 1000, 2),
                    )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                if logger:
                    logger.error(
                        f"{func.__name__} failed",
                        duration_ms=round(elapsed * 1000, 2),
                        error=str(exc),
                    )
                raise
        return wrapper
    return decorator
