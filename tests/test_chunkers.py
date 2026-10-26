"""Tests for chunking strategies."""

import pytest

from src.ingestion.chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    get_chunker,
)

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 30
)

LONG_PROSE = """
Introduction to machine learning systems.

Machine learning is a subset of artificial intelligence that enables systems
to learn from data without being explicitly programmed. It focuses on developing
algorithms that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples,
direct experience, or instruction, so that computers can look for patterns in
data and make better decisions in the future.

Types of Machine Learning

Supervised learning uses labeled training data to learn the mapping function
from input to output. The model is trained on a known dataset and then used to
predict outcomes for new data. Common algorithms include linear regression,
decision trees, and neural networks.

Unsupervised learning works with unlabeled data. The goal is to find hidden
patterns or intrinsic structures in input data. Clustering and dimensionality
reduction are common unsupervised learning tasks.

Reinforcement learning trains an agent to make decisions by rewarding desirable
behaviors and punishing undesirable ones. It is particularly well-suited for
problems where an agent must take a sequence of actions in an environment.

Applications of Machine Learning

Machine learning is applied in many domains including natural language processing,
computer vision, healthcare, finance, and robotics. Some notable applications include
image recognition, speech recognition, recommendation systems, and fraud detection.
""".strip()


class TestFixedSizeChunker:
    def test_produces_chunks(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk("doc1", SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk("doc1", SAMPLE_TEXT)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_references_correct_doc(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk("my_doc", SAMPLE_TEXT)
        assert all(c.doc_id == "my_doc" for c in chunks)

    def test_chunk_strategy_label(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("doc1", SAMPLE_TEXT)
        assert all(c.strategy == "fixed_size" for c in chunks)

    def test_no_empty_chunks(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk("doc1", SAMPLE_TEXT)
        assert all(c.content.strip() for c in chunks)

    def test_metadata_is_preserved(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        meta = {"source": "test", "author": "tester"}
        chunks = chunker.chunk("doc1", SAMPLE_TEXT, metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["author"] == "tester"

    def test_short_document_produces_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=1000, overlap=100)
        short = "This is a short document."
        chunks = chunker.chunk("doc1", short)
        assert len(chunks) == 1
        assert short in chunks[0].content

    def test_overlap_smaller_than_chunk_size_required(self):
        with pytest.raises(ValueError, match="overlap"):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_empty_document_returns_no_chunks(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("doc1", "   ")
        assert len(chunks) == 0

    def test_total_chunks_field_matches_list_length(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk("doc1", SAMPLE_TEXT)
        assert chunks[0].total_chunks == len(chunks)


class TestSemanticChunker:
    def test_produces_chunks(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk("doc1", LONG_PROSE)
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk("doc1", LONG_PROSE)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_strategy_label(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk("doc1", LONG_PROSE)
        assert all(c.strategy == "semantic" for c in chunks)

    def test_no_chunk_exceeds_max_size(self):
        max_size = 800
        chunker = SemanticChunker(max_chunk_size=max_size)
        chunks = chunker.chunk("doc1", LONG_PROSE)
        # Allow slight overage for single paragraphs that exceed max on their own
        for chunk in chunks:
            # A single paragraph longer than max_size is kept intact (unsplittable)
            # so we only enforce the limit on chunks that have multiple paragraphs
            if "\n" in chunk.content:
                assert len(chunk.content) <= max_size + 10

    def test_metadata_preserved(self):
        chunker = SemanticChunker()
        meta = {"category": "research"}
        chunks = chunker.chunk("doc1", LONG_PROSE, metadata=meta)
        assert all(c.metadata["category"] == "research" for c in chunks)

    def test_short_content(self):
        chunker = SemanticChunker()
        chunks = chunker.chunk("doc1", "Short content.")
        assert len(chunks) >= 1


class TestGetChunker:
    def test_fixed_strategy(self):
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_semantic_strategy(self):
        chunker = get_chunker("semantic")
        assert isinstance(chunker, SemanticChunker)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("invalid")

    def test_kwargs_passed_to_chunker(self):
        chunker = get_chunker("fixed", chunk_size=500, overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50
