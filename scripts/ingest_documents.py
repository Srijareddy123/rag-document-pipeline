#!/usr/bin/env python3
"""
Standalone ingestion script for local testing.

Usage:
    python scripts/ingest_documents.py                        # ingest all sample docs
    python scripts/ingest_documents.py --dir data/samples     # specific directory
    python scripts/ingest_documents.py --strategy semantic    # change chunking

Run this after `docker compose up` to populate ChromaDB with sample documents,
then hit the /api/v1/query endpoint to test retrieval.
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.factory import get_embedding_provider
from src.ingestion.pipeline import DocumentIngestionPipeline
from src.monitoring.logger import get_logger
from src.storage.chroma import ChromaVectorStore

logger = get_logger("ingest_script")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--dir",
        default="data/samples",
        help="Directory containing .txt files to ingest (default: data/samples)",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic", "recursive"],
        default="fixed",
        help="Chunking strategy (default: fixed)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="ChromaDB collection name (default: from env or 'documents')",
    )
    args = parser.parse_args()

    doc_dir = Path(args.dir)
    if not doc_dir.exists():
        logger.error(f"Directory not found: {doc_dir}")
        sys.exit(1)

    files = list(doc_dir.glob("*.txt"))
    if not files:
        logger.error(f"No .txt files found in {doc_dir}")
        sys.exit(1)

    logger.info(f"Found {len(files)} files in {doc_dir}")
    for f in files:
        logger.info(f"  {f.name} ({f.stat().st_size:,} bytes)")

    embedder = get_embedding_provider()
    store = ChromaVectorStore(collection_name=args.collection)

    pipeline = DocumentIngestionPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunk_strategy=args.strategy,
    )

    logger.info(f"Starting ingestion with '{args.strategy}' chunking strategy...")
    result = pipeline.ingest_files(files)

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"  Total documents:    {result.total_documents}")
    print(f"  Accepted:           {result.accepted_documents}")
    print(f"  Rejected:           {result.rejected_documents}")
    print(f"  Total chunks:       {result.total_chunks}")
    print(f"  Duration:           {result.duration_seconds}s")
    print(f"  Accepted rate:      {result.success_rate:.1%}")

    if result.failed_doc_ids:
        print(f"\n  Failed doc IDs: {result.failed_doc_ids}")

    if result.rejected_documents > 0:
        print("\n  Rejected documents:")
        for report in result.quality_reports:
            if not report.passed:
                print(f"    - {report.doc_id}: {report.reason}")

    print(f"\n  ChromaDB now contains {store.count()} total chunks.")
    print("  Run the API server and POST to /api/v1/query to test retrieval.")


if __name__ == "__main__":
    main()
