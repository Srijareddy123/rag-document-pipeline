#!/usr/bin/env python3
"""
Load test: ingest N articles of Simple English Wikipedia and report the
document count, chunk count, wall time, throughput and peak memory.

    python scripts/fetch_corpus.py
    python scripts/load_test.py --n 50000 --strategy fixed

The vector store lives in --chroma-dir (default: a temp folder next to the
corpus, outside the repo). Results are written to results/load_test.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "WARNING")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from scripts.fetch_corpus import corpus_path  # noqa: E402
from src.embeddings.factory import get_embedding_provider  # noqa: E402
from src.ingestion.pipeline import DocumentIngestionPipeline  # noqa: E402
from src.storage.chroma import ChromaVectorStore  # noqa: E402

READ_BATCH = 1000
N_QUERIES = 100


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return peak / 1e6 if platform.system() == "Darwin" else peak / 1e3


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50_000, help="number of articles")
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "semantic", "recursive"])
    parser.add_argument("--corpus", default=None, help="parquet file (default: fetch_corpus path)")
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument("--out", default="results/load_test.json")
    args = parser.parse_args()

    corpus = Path(args.corpus) if args.corpus else corpus_path()
    if not corpus.exists():
        print(f"Corpus not found at {corpus}. Run: python scripts/fetch_corpus.py")
        return 1
    chroma_dir = Path(args.chroma_dir or corpus.parent / "load_test_chroma")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    embedder = get_embedding_provider("local")
    store = ChromaVectorStore(collection_name="load_test", persist_directory=str(chroma_dir))
    pipeline = DocumentIngestionPipeline(embedder, store, chunk_strategy=args.strategy)

    titles: list[str] = []
    seen = accepted = rejected = chunks = 0
    embed_s = store_s = 0.0
    text_chars = 0
    reasons: collections.Counter[str] = collections.Counter()
    start = time.perf_counter()

    for batch in pq.ParquetFile(corpus).iter_batches(
        batch_size=READ_BATCH, columns=["id", "title", "text"]
    ):
        rows = batch.to_pylist()[: args.n - seen]
        if not rows:
            break
        docs = [(f"wiki_{r['id']}", r["text"]) for r in rows]
        meta = [{"title": r["title"]} for r in rows]
        result = pipeline.ingest_texts(docs, meta, keep_passed_reports=False)
        seen += len(rows)
        accepted += result.accepted_documents
        rejected += result.rejected_documents
        chunks += result.total_chunks
        embed_s += result.embed_seconds
        store_s += result.store_seconds
        for rep in result.quality_reports:
            if not rep.passed:
                reasons.update(rep.reason.split(", "))
        text_chars += sum(len(t) for _, t in docs)
        titles.extend(r["title"] for r in rows[::10])
        elapsed = time.perf_counter() - start
        print(
            f"  {seen:>6} docs  {chunks:>7} chunks  {elapsed:7.0f}s  "
            f"peak {peak_rss_mb():.0f} MB",
            flush=True,
        )
        if seen >= args.n:
            break

    wall = time.perf_counter() - start

    step = max(1, len(titles) // N_QUERIES)
    store.query(embedder.embed_query("warm up"), n_results=5)
    latencies = []
    for title in titles[::step][:N_QUERIES]:
        t0 = time.perf_counter()
        store.query(embedder.embed_query(f"What is {title}?"), n_results=5)
        latencies.append((time.perf_counter() - t0) * 1000)

    report = {
        "command": f"/usr/bin/time -l python scripts/load_test.py --n {args.n} --strategy {args.strategy}",
        "corpus": "Simple English Wikipedia, 20231101 (wikimedia/wikipedia on Hugging Face)",
        "strategy": args.strategy,
        "embedding_model": getattr(embedder, "_model_name", "unknown"),
        "documents_read": seen,
        "documents_accepted": accepted,
        "documents_rejected": rejected,
        "rejected_by_check": dict(reasons),
        "chunks_stored": chunks,
        "chunks_in_store": store.count(),
        "text_megabytes": round(text_chars / 1e6, 1),
        "wall_seconds": round(wall, 1),
        "docs_per_second": round(seen / wall, 1),
        "chunks_per_second": round(chunks / wall, 1),
        "embed_seconds": round(embed_s, 1),
        "store_seconds": round(store_s, 1),
        "peak_rss_mb": round(peak_rss_mb()),
        "query_latency_ms_p50": round(percentile(latencies, 0.5), 1),
        "query_latency_ms_p95": round(percentile(latencies, 0.95), 1),
        "queries_timed": len(latencies),
        "machine": f"{platform.machine()} {platform.system()}, Python {platform.python_version()}",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
