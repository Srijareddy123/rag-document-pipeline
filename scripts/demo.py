#!/usr/bin/env python3
"""
Demo: runs the whole pipeline on the sample documents, offline, with no API keys.

    make demo

Loads the files in data/samples, rejects the bad ones, chunks the rest two ways,
embeds with a local sentence-transformers model (downloaded once, about 90 MB),
stores in a temporary ChromaDB folder and answers a few questions.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.factory import get_embedding_provider  # noqa: E402
from src.ingestion.pipeline import DocumentIngestionPipeline  # noqa: E402
from src.storage.chroma import ChromaVectorStore  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
QUESTIONS = [
    "How does HNSW indexing work?",
    "How long do we keep customer data?",
    "What does retrieval-augmented generation combine?",
]


def snippet(text: str, width: int = 90) -> str:
    return " ".join(text.split())[:width]


def main() -> int:
    files = sorted(SAMPLES.glob("*.txt"))
    embedder = get_embedding_provider("local")
    print(f"Embedding provider: local ({embedder._model_name}), no API keys needed\n")

    with tempfile.TemporaryDirectory() as tmp:
        for strategy in ("fixed", "semantic"):
            store = ChromaVectorStore(collection_name=strategy, persist_directory=tmp)
            pipeline = DocumentIngestionPipeline(embedder, store, chunk_strategy=strategy)
            result = pipeline.ingest_files(files)
            print(
                f"[{strategy}] {result.total_documents} files, "
                f"{result.accepted_documents} accepted, {result.rejected_documents} rejected, "
                f"{result.total_chunks} chunks, {result.duration_seconds}s"
            )
            if strategy == "fixed":
                for report in result.quality_reports:
                    if not report.passed:
                        print(f"  rejected {report.doc_id}: {report.reason}")
            for question in QUESTIONS:
                hit = store.query(embedder.embed_query(question), n_results=1)[0]
                print(f"  Q: {question}")
                print(f"     {hit['id']}  score {hit['score']}  {snippet(hit['document'])}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
