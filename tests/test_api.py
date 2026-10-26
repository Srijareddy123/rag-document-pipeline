"""API tests with FastAPI's TestClient, an in-memory ChromaDB and a hashing embedder."""

from __future__ import annotations

import uuid

import chromadb
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import get_embedder, get_store
from src.storage.chroma import ChromaVectorStore
from tests.test_samples_and_eval import HashingEmbedder

DOC = "Vector databases store embeddings and answer nearest neighbour queries quickly. " * 5


@pytest.fixture
def client():
    store = ChromaVectorStore(collection_name=f"api_{uuid.uuid4().hex}")
    store._client = chromadb.EphemeralClient()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: HashingEmbedder()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_reports_empty_store(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["vector_store_count"] == 0


def test_query_before_ingest_is_404(client):
    r = client.post("/api/v1/query", json={"query": "anything"})
    assert r.status_code == 404


def test_ingest_then_query_round_trip(client):
    r = client.post(
        "/api/v1/ingest",
        json={"documents": [{"id": "d1", "content": DOC}], "chunk_strategy": "semantic"},
    )
    assert r.status_code == 200
    assert r.json()["accepted_documents"] == 1

    q = client.post("/api/v1/query", json={"query": "nearest neighbour embeddings", "n_results": 2})
    assert q.status_code == 200
    assert q.json()["results"][0]["id"].startswith("d1_")


def test_ingest_rejects_unknown_strategy_and_bad_metadata(client):
    docs = [{"id": "d1", "content": DOC}]
    assert client.post(
        "/api/v1/ingest", json={"documents": docs, "chunk_strategy": "nope"}
    ).status_code == 422
    assert client.post(
        "/api/v1/ingest", json={"documents": docs, "metadata": [{}, {}]}
    ).status_code == 422


def test_ingest_accepts_recursive_strategy_in_schema(client):
    r = client.post(
        "/api/v1/ingest",
        json={"documents": [{"id": "d2", "content": DOC}], "chunk_strategy": "recursive"},
    )
    assert r.status_code == 200
