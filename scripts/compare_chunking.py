#!/usr/bin/env python3
"""
Compare chunking strategies with a small retrieval measurement.

Indexes the first --docs articles of the corpus once per strategy, then asks
--queries sentence-based questions (see src/evaluation.py) and reports hit
rates and MRR. Results go to results/chunking_comparison.json.

    python scripts/fetch_corpus.py
    python scripts/compare_chunking.py --docs 2000 --queries 300
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from scripts.fetch_corpus import corpus_path  # noqa: E402
from src.embeddings.factory import get_embedding_provider  # noqa: E402
from src.evaluation import build_queries, doc_hit, rank_of_evidence, summarize  # noqa: E402
from src.ingestion.chunkers import get_chunker  # noqa: E402
from src.ingestion.pipeline import DocumentIngestionPipeline  # noqa: E402
from src.storage.chroma import ChromaVectorStore  # noqa: E402

STRATEGIES = ("fixed", "semantic", "recursive")
TOP_K = 5


def load_docs(path: Path, n: int) -> list[tuple[str, str]]:
    table = pq.ParquetFile(path).iter_batches(batch_size=n, columns=["id", "text"])
    rows = next(iter(table)).to_pylist()[:n]
    return [(f"wiki_{r['id']}", r["text"]) for r in rows]


def mean_chunk_chars(strategy: str, docs: list[tuple[str, str]]) -> int:
    chunker = get_chunker(strategy)
    sizes = [len(c.content) for d, t in docs for c in chunker.chunk(d, t)]
    return round(sum(sizes) / max(len(sizes), 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--out", default="results/chunking_comparison.json")
    args = parser.parse_args()

    corpus = corpus_path()
    if not corpus.exists():
        print(f"Corpus not found at {corpus}. Run: python scripts/fetch_corpus.py")
        return 1

    docs = load_docs(corpus, args.docs)
    queries = build_queries(docs, args.queries)
    embedder = get_embedding_provider("local")
    work = corpus.parent / "compare_chunking_chroma"
    if work.exists():
        shutil.rmtree(work)

    report: dict = {
        "command": f"python scripts/compare_chunking.py --docs {args.docs} --queries {args.queries}",
        "corpus": "Simple English Wikipedia, 20231101",
        "documents_indexed": len(docs),
        "queries": len(queries),
        "top_k": TOP_K,
        "embedding_model": getattr(embedder, "_model_name", "unknown"),
        "strategies": {},
    }

    for strategy in STRATEGIES:
        store = ChromaVectorStore(collection_name=strategy, persist_directory=str(work))
        pipeline = DocumentIngestionPipeline(embedder, store, chunk_strategy=strategy)
        t0 = time.perf_counter()
        result = pipeline.ingest_texts(docs, keep_passed_reports=False)
        ingest_s = time.perf_counter() - t0

        entry = {
            "chunks": result.total_chunks,
            "mean_chunk_chars": mean_chunk_chars(strategy, docs),
            "ingest_seconds": round(ingest_s, 1),
        }
        for mode in ("verbatim", "dropout"):
            ranks, doc_hits = [], []
            for q in queries:
                text = getattr(q, mode)
                hits = store.query(embedder.embed_query(text), n_results=TOP_K)
                ranks.append(rank_of_evidence(hits, q))
                doc_hits.append(doc_hit(hits, q))
            entry[mode] = summarize(ranks, doc_hits)
        report["strategies"][strategy] = entry
        print(strategy, json.dumps(entry), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
