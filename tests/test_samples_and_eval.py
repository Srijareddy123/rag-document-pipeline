"""
Tests on the committed sample data and the evaluation helpers.

The hashing embedder is deterministic and needs no model download. One test at
the end uses the real local model; it is skipped when sentence-transformers
cannot load it (for example offline on a fresh machine).
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import uuid
from pathlib import Path

import chromadb
import pytest

from src.embeddings.factory import get_embedding_provider
from src.evaluation import (
    EvalQuery,
    build_queries,
    coverage,
    doc_hit,
    drop_words,
    rank_of_evidence,
    summarize,
)
from src.ingestion.chunkers import STRATEGIES, get_chunker
from src.ingestion.pipeline import DocumentIngestionPipeline
from src.storage.chroma import ChromaVectorStore

SAMPLES = sorted(Path(__file__).resolve().parent.parent.joinpath("data/samples").glob("*.txt"))
DIM = 256


class HashingEmbedder:
    """Bag-of-words hashing embedder: same words give similar vectors."""

    def embed_texts(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)

    @staticmethod
    def _one(text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            vec[int.from_bytes(digest[:2], "big") % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def make_store() -> ChromaVectorStore:
    store = ChromaVectorStore(collection_name=f"t_{uuid.uuid4().hex}")
    store._client = chromadb.EphemeralClient()
    return store


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_samples_ingest_and_bad_files_are_rejected(strategy):
    pipeline = DocumentIngestionPipeline(HashingEmbedder(), make_store(), chunk_strategy=strategy)
    result = pipeline.ingest_files(SAMPLES)
    assert result.total_documents == len(SAMPLES)
    rejected = {r.doc_id: r.reason for r in result.quality_reports if not r.passed}
    assert "sample_empty" in rejected
    assert "not_duplicate" in " ".join(rejected.values())
    assert result.accepted_documents == len(SAMPLES) - len(rejected)
    assert result.total_chunks > 0


def test_sample_question_finds_the_right_document():
    store = make_store()
    embedder = HashingEmbedder()
    DocumentIngestionPipeline(embedder, store, chunk_strategy="semantic").ingest_files(SAMPLES)
    hit = store.query(embedder.embed_query("how long do we keep customer data retention"), 1)[0]
    assert hit["id"].startswith("sample_policy")


def test_small_document_batches_give_the_same_totals():
    docs = [(f"d{i}", f"Document number {i} talks about vector search and retrieval. " * 8)
            for i in range(25)]
    totals = []
    for batch in (1, 7, 200):
        pipeline = DocumentIngestionPipeline(
            HashingEmbedder(), make_store(), doc_batch_size=batch
        )
        r = pipeline.ingest_texts(docs)
        totals.append((r.accepted_documents, r.rejected_documents, r.total_chunks))
    assert totals[0] == totals[1] == totals[2]


def test_duplicates_are_caught_across_batches():
    text = "The same paragraph about embeddings and vector stores, repeated for length. " * 5
    pipeline = DocumentIngestionPipeline(HashingEmbedder(), make_store(), doc_batch_size=1)
    result = pipeline.ingest_texts([("a", text), ("b", text)])
    assert (result.accepted_documents, result.rejected_documents) == (1, 1)


def test_recursive_chunker_respects_size_and_ids():
    chunker = get_chunker("recursive", chunk_size=300, overlap=50)
    text = "Sentence about retrieval. " * 100
    chunks = chunker.chunk("doc", text)
    assert len(chunks) > 1
    assert all(len(c.content) <= 300 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_unknown_strategy_lists_the_valid_ones():
    with pytest.raises(ValueError, match="recursive"):
        get_chunker("nope")


def test_unknown_embedding_provider_is_rejected():
    with pytest.raises(ValueError):
        get_embedding_provider("nope")


def test_coverage_and_drop_words():
    assert coverage("alpha beta gamma delta", "beta alpha gamma delta epsilon") == 1.0
    assert coverage("alpha beta gamma delta", "alpha beta") == 0.5
    assert coverage("", "anything") == 0.0
    out = drop_words("one two three four five six seven eight", random.Random(1))
    assert 0 < len(out.split()) < 8


def test_build_queries_is_seeded_and_uses_long_documents():
    long_text = " ".join(
        f"This is sentence number {i} about a particular topic in the document body." for i in range(60)
    )
    docs = [("short", "tiny text."), ("long1", long_text), ("long2", long_text + " extra.")]
    first = build_queries(docs, 5, seed=3)
    again = build_queries(docs, 5, seed=3)
    assert first == again
    assert {q.doc_id for q in first} == {"long1", "long2"}


def test_rank_of_evidence_and_summary():
    q = EvalQuery("doc1", "alpha beta gamma delta epsilon", "x", "y")
    hits = [
        {"id": "doc2_chunk_0000", "document": "alpha beta gamma delta epsilon"},
        {"id": "doc1_chunk_0001", "document": "alpha beta"},
        {"id": "doc1_chunk_0002", "document": "alpha beta gamma delta epsilon zeta"},
    ]
    assert rank_of_evidence(hits, q) == 3
    assert doc_hit(hits, q) is True
    assert rank_of_evidence(hits[:2], q) is None
    summary = summarize([3, None, 1], [True, True, False])
    assert summary["evidence_hit_at_1"] == 0.333
    assert summary["evidence_hit_at_5"] == 0.667
    assert summary["evidence_mrr_at_5"] == 0.444


def test_real_local_model_ranks_the_relevant_sample_first():
    try:
        embedder = get_embedding_provider("local")
        embedder.embed_query("warm up")
    except Exception as exc:  # model not downloadable here
        pytest.skip(f"local model unavailable: {exc}")
    store = make_store()
    DocumentIngestionPipeline(embedder, store, chunk_strategy="semantic").ingest_files(SAMPLES)
    hit = store.query(embedder.embed_query("How does HNSW indexing work?"), 1)[0]
    assert hit["id"].startswith("sample_tech_doc")
