"""
Integration tests for the ingestion pipeline.

Uses ChromaDB in-memory mode (EphemeralClient) and a stub embedding provider
so tests run fully offline with no disk I/O and no model downloads.
"""

from __future__ import annotations

from src.ingestion.pipeline import DocumentIngestionPipeline

GOOD_DOC = (
    "This is a well-formed document with sufficient content to pass quality checks. "
    "It discusses machine learning, natural language processing, and vector databases. " * 4
)

GOOD_DOC_2 = (
    "A second document covering entirely different topics such as distributed systems, "
    "database indexing strategies, and cloud infrastructure patterns. " * 4
)


class StubEmbeddingProvider:
    """Returns deterministic dummy embeddings. No ML, no network."""

    DIM = 8

    def embed_texts(self, texts):
        return [[float(i % self.DIM) / self.DIM for i in range(self.DIM)] for _ in texts]

    def embed_query(self, text):
        return self.embed_texts([text])[0]

    @property
    def dimension(self):
        return self.DIM


def make_chroma_store(collection_name: str = "test_col"):
    """Returns an ephemeral ChromaDB store, no disk writes, no persistence."""
    import chromadb

    from src.storage.chroma import ChromaVectorStore

    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name=collection_name)
    # Inject the ephemeral client so no disk access occurs
    store._client = client
    return store


def make_pipeline(strategy: str = "fixed") -> DocumentIngestionPipeline:
    return DocumentIngestionPipeline(
        embedding_provider=StubEmbeddingProvider(),
        vector_store=make_chroma_store(),
        chunk_strategy=strategy,
    )


class TestIngestionPipeline:
    def test_single_good_document_ingested(self):
        pipeline = make_pipeline()
        result = pipeline.ingest_texts([("doc1", GOOD_DOC)])
        assert result.accepted_documents == 1
        assert result.rejected_documents == 0
        assert result.total_chunks > 0

    def test_multiple_good_documents_ingested(self):
        pipeline = make_pipeline()
        docs = [("doc1", GOOD_DOC), ("doc2", GOOD_DOC_2)]
        result = pipeline.ingest_texts(docs)
        assert result.accepted_documents == 2
        assert result.total_chunks > 0

    def test_empty_document_rejected(self):
        pipeline = make_pipeline()
        result = pipeline.ingest_texts([("empty", "")])
        assert result.rejected_documents == 1
        assert result.accepted_documents == 0
        assert result.total_chunks == 0

    def test_mixed_batch_partial_rejection(self):
        pipeline = make_pipeline()
        docs = [
            ("good", GOOD_DOC),
            ("bad_empty", ""),
            ("bad_short", "Too short."),
            ("good2", GOOD_DOC_2),
        ]
        result = pipeline.ingest_texts(docs)
        assert result.accepted_documents == 2
        assert result.rejected_documents == 2

    def test_result_has_quality_reports(self):
        pipeline = make_pipeline()
        result = pipeline.ingest_texts([("doc1", GOOD_DOC)])
        assert len(result.quality_reports) == 1
        assert result.quality_reports[0].passed is True

    def test_success_rate_calculation(self):
        pipeline = make_pipeline()
        docs = [("good", GOOD_DOC), ("bad", "")]
        result = pipeline.ingest_texts(docs)
        # 1 accepted out of 2 total
        assert result.success_rate == 0.5

    def test_duration_is_non_negative(self):
        pipeline = make_pipeline()
        result = pipeline.ingest_texts([("doc1", GOOD_DOC)])
        assert result.duration_seconds >= 0

    def test_total_documents_matches_input(self):
        pipeline = make_pipeline()
        docs = [("a", GOOD_DOC), ("b", GOOD_DOC_2)]
        result = pipeline.ingest_texts(docs)
        assert result.total_documents == 2

    def test_semantic_strategy_works(self):
        pipeline = make_pipeline(strategy="semantic")
        result = pipeline.ingest_texts([("doc1", GOOD_DOC)])
        assert result.accepted_documents == 1
        assert result.total_chunks > 0

    def test_metadata_passed_through(self):
        store = make_chroma_store()
        pipeline = DocumentIngestionPipeline(
            embedding_provider=StubEmbeddingProvider(),
            vector_store=store,
            chunk_strategy="fixed",
        )
        meta = [{"source": "unit_test", "category": "engineering"}]
        pipeline.ingest_texts([("doc1", GOOD_DOC)], metadatas=meta)

        count = store.count()
        assert count > 0

    def test_embedding_failure_tracked_in_failed_ids(self):
        class FailingEmbedder:
            def embed_texts(self, texts):
                raise RuntimeError("Simulated embedding failure")
            def embed_query(self, text):
                raise RuntimeError("Simulated embedding failure")
            @property
            def dimension(self):
                return 8

        store = make_chroma_store()
        pipeline = DocumentIngestionPipeline(
            embedding_provider=FailingEmbedder(),
            vector_store=store,
            chunk_strategy="fixed",
        )
        result = pipeline.ingest_texts([("doc1", GOOD_DOC)])
        assert "doc1" in result.failed_doc_ids

    def test_empty_input_returns_zero_result(self):
        pipeline = make_pipeline()
        result = pipeline.ingest_texts([])
        assert result.total_documents == 0
        assert result.total_chunks == 0

    def test_ingest_files_loads_sample_docs(self):
        from pathlib import Path

        store = make_chroma_store()
        pipeline = DocumentIngestionPipeline(
            embedding_provider=StubEmbeddingProvider(),
            vector_store=store,
            chunk_strategy="fixed",
        )

        sample_dir = Path("data/samples")
        files = list(sample_dir.glob("*.txt"))
        assert len(files) > 0, "No sample files found, check data/samples/"

        result = pipeline.ingest_files(files)
        assert result.accepted_documents > 0
        assert result.total_chunks > 0
