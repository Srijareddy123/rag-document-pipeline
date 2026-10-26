"""
Tests for vector store query and retrieval behavior.

Uses ChromaDB EphemeralClient so nothing is written to disk.
Each test uses a unique collection name to avoid cross-test contamination -
ChromaDB 0.5.20 shares EphemeralClient state within a process.
"""

from __future__ import annotations

import uuid

import chromadb

from src.storage.chroma import ChromaVectorStore

DIM = 8


def _make_store() -> ChromaVectorStore:
    # Unique collection name per test to prevent cross-test state leakage.
    # Discovered during test development: EphemeralClient shares in-process state
    # in ChromaDB 0.5.x even across separate client instances.
    store = ChromaVectorStore(collection_name=f"test_{uuid.uuid4().hex}")
    store._client = chromadb.EphemeralClient()
    return store


def _unit_vec(index: int, dim: int = DIM) -> list[float]:
    """Returns a unit vector with 1.0 at position `index % dim`."""
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    return vec


class TestChromaVectorStore:
    def test_empty_store_count_is_zero(self):
        store = _make_store()
        assert store.count() == 0

    def test_upsert_increases_count(self):
        store = _make_store()
        store.upsert(
            ids=["a", "b"],
            embeddings=[_unit_vec(0), _unit_vec(1)],
            documents=["doc a", "doc b"],
        )
        assert store.count() == 2

    def test_query_returns_results(self):
        store = _make_store()
        store.upsert(
            ids=["chunk_0"],
            embeddings=[_unit_vec(0)],
            documents=["retrieval test document"],
        )
        results = store.query(query_embedding=_unit_vec(0), n_results=1)
        assert len(results) == 1
        assert results[0]["id"] == "chunk_0"

    def test_query_returns_closest_first(self):
        store = _make_store()
        # chunk_near aligns with query (dim 0); chunk_far is orthogonal (dim 1)
        store.upsert(
            ids=["chunk_near", "chunk_far"],
            embeddings=[_unit_vec(0), _unit_vec(1)],
            documents=["near doc", "far doc"],
        )
        results = store.query(query_embedding=_unit_vec(0), n_results=2)
        assert results[0]["id"] == "chunk_near"

    def test_score_between_zero_and_one(self):
        store = _make_store()
        store.upsert(
            ids=["x"],
            embeddings=[_unit_vec(0)],
            documents=["content"],
        )
        results = store.query(query_embedding=_unit_vec(0), n_results=1)
        score = results[0]["score"]
        assert 0.0 <= score <= 1.0

    def test_n_results_respected(self):
        store = _make_store()
        for i in range(10):
            store.upsert(
                ids=[f"doc_{i}"],
                embeddings=[_unit_vec(i)],
                documents=[f"document {i}"],
            )
        results = store.query(query_embedding=_unit_vec(0), n_results=3)
        assert len(results) == 3

    def test_n_results_capped_at_collection_size(self):
        store = _make_store()
        store.upsert(
            ids=["only_one"],
            embeddings=[_unit_vec(0)],
            documents=["single doc"],
        )
        results = store.query(query_embedding=_unit_vec(0), n_results=10)
        assert len(results) == 1

    def test_empty_store_returns_empty_list(self):
        store = _make_store()
        results = store.query(query_embedding=_unit_vec(0), n_results=5)
        assert results == []

    def test_upsert_updates_existing_id(self):
        store = _make_store()
        store.upsert(
            ids=["doc1"],
            embeddings=[_unit_vec(0)],
            documents=["original content"],
        )
        store.upsert(
            ids=["doc1"],
            embeddings=[_unit_vec(0)],
            documents=["updated content"],
        )
        assert store.count() == 1
        results = store.query(query_embedding=_unit_vec(0), n_results=1)
        assert results[0]["document"] == "updated content"

    def test_metadata_returned_in_results(self):
        store = _make_store()
        store.upsert(
            ids=["doc1"],
            embeddings=[_unit_vec(0)],
            documents=["content"],
            metadatas=[{"source": "test_suite", "category": "unit"}],
        )
        results = store.query(query_embedding=_unit_vec(0), n_results=1)
        assert results[0]["metadata"]["source"] == "test_suite"
        assert results[0]["metadata"]["category"] == "unit"

    def test_upsert_empty_list_is_noop(self):
        store = _make_store()
        store.upsert(ids=[], embeddings=[], documents=[])
        assert store.count() == 0

    def test_delete_collection_resets_count(self):
        store = _make_store()
        store.upsert(
            ids=["x"],
            embeddings=[_unit_vec(0)],
            documents=["something"],
        )
        assert store.count() == 1
        store.delete_collection()
        assert store.count() == 0

    def test_large_batch_upsert_succeeds(self):
        """Verify the 5000-item batch limit is handled transparently."""
        store = _make_store()
        n = 6000
        ids = [f"chunk_{i}" for i in range(n)]
        embeddings = [_unit_vec(i) for i in range(n)]
        documents = [f"Document number {i}" for i in range(n)]
        store.upsert(ids=ids, embeddings=embeddings, documents=documents)
        assert store.count() == n
