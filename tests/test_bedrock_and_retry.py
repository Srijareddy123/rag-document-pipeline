"""Bedrock provider and retry decorator, tested with a fake client (no AWS calls)."""

from __future__ import annotations

import io
import json

import pytest

from src.embeddings.bedrock import BEDROCK_EMBED_DIM, BedrockEmbeddingProvider
from src.embeddings.factory import get_embedding_provider
from src.ingestion.retry import with_retry


class FakeBedrockClient:
    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first

    def invoke_model(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("ThrottlingException: slow down")
        body = json.loads(kwargs["body"])
        vec = [float(len(body["inputText"]))] * BEDROCK_EMBED_DIM
        return {"body": io.BytesIO(json.dumps({"embedding": vec}).encode())}


def make_provider(client) -> BedrockEmbeddingProvider:
    provider = BedrockEmbeddingProvider(region="us-east-1")
    provider._client = client
    return provider


def test_embed_texts_returns_one_vector_per_text_and_zero_vector_for_blank():
    provider = make_provider(FakeBedrockClient())
    vectors = provider.embed_texts(["abc", "   ", "hello"])
    assert [len(v) for v in vectors] == [BEDROCK_EMBED_DIM] * 3
    assert vectors[0][0] == 3.0
    assert vectors[1] == [0.0] * BEDROCK_EMBED_DIM
    assert provider.dimension == BEDROCK_EMBED_DIM
    assert provider.embed_texts([]) == []


def test_throttled_call_is_retried(monkeypatch):
    monkeypatch.setattr("src.ingestion.retry.time.sleep", lambda _: None)
    client = FakeBedrockClient(fail_first=2)
    vector = make_provider(client).embed_query("retry me")
    assert len(vector) == BEDROCK_EMBED_DIM
    assert client.calls == 3


def test_factory_returns_bedrock_when_asked():
    assert isinstance(get_embedding_provider("bedrock"), BedrockEmbeddingProvider)


def test_with_retry_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("src.ingestion.retry.time.sleep", lambda _: None)
    calls = []

    @with_retry(max_attempts=3, base_delay=0.0)
    def always_fails() -> None:
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        always_fails()
    assert len(calls) == 3
