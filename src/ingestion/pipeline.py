"""
Core ingestion pipeline.

Orchestrates: quality check, chunk, embed, store.

Documents are processed in batches (DOC_BATCH_SIZE documents at a time), so
peak memory depends on the batch size and not on the corpus size. A document
that fails to chunk, embed or store is recorded in `failed_doc_ids` and the
run continues.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.chunkers import Chunk, get_chunker
from src.ingestion.quality import DocumentQualityChecker, QualityReport
from src.monitoring.logger import StructuredLogger

logger = StructuredLogger(__name__)

DEFAULT_CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "fixed")
DEFAULT_EMBED_BATCH = int(os.getenv("EMBED_BATCH_SIZE", "32"))
DEFAULT_DOC_BATCH = int(os.getenv("DOC_BATCH_SIZE", "200"))


@dataclass
class IngestionResult:
    total_documents: int
    accepted_documents: int
    rejected_documents: int
    total_chunks: int
    duration_seconds: float
    failed_doc_ids: list[str] = field(default_factory=list)
    quality_reports: list[QualityReport] = field(default_factory=list)
    embed_seconds: float = 0.0
    store_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return round(self.accepted_documents / self.total_documents, 4)


class DocumentIngestionPipeline:
    """
    End-to-end ingestion: quality check -> chunk -> embed -> store.

    Parameters
    ----------
    embedding_provider : provider with embed_texts(texts) -> list[list[float]]
    vector_store       : store with upsert(ids, embeddings, documents, metadatas)
    chunk_strategy     : 'fixed', 'semantic' or 'recursive'
    chunk_kwargs       : passed to the chunker constructor
    embed_batch_size   : number of chunks per embedding call
    """

    def __init__(
        self,
        embedding_provider,
        vector_store,
        chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
        chunk_kwargs: dict | None = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH,
        doc_batch_size: int = DEFAULT_DOC_BATCH,
    ):
        self._embedder = embedding_provider
        self._store = vector_store
        self._chunker = get_chunker(chunk_strategy, **(chunk_kwargs or {}))
        self._embed_batch_size = embed_batch_size
        self._doc_batch_size = max(1, doc_batch_size)
        self._embed_seconds = 0.0
        self._store_seconds = 0.0
        self._quality = DocumentQualityChecker()

    def ingest_texts(
        self,
        documents: Sequence[tuple[str, str]],
        metadatas: Sequence[dict] | None = None,
        keep_passed_reports: bool = True,
    ) -> IngestionResult:
        """
        Ingest a list of (doc_id, content) pairs.

        Parameters
        ----------
        documents : list of (doc_id, content) tuples
        metadatas : optional per-document metadata dicts
        keep_passed_reports : set False on very large runs to keep only the
            quality reports of rejected documents
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]

        start = time.perf_counter()
        self._embed_seconds = 0.0
        self._store_seconds = 0.0
        accepted_total = 0
        rejected_total = 0
        chunk_total = 0
        failed_doc_ids: list[str] = []
        reports: list[QualityReport] = []

        for lo in range(0, len(documents), self._doc_batch_size):
            batch = list(documents[lo : lo + self._doc_batch_size])
            meta_by_id = {
                doc_id: meta
                for (doc_id, _), meta in zip(batch, metadatas[lo : lo + self._doc_batch_size])
            }
            accepted, batch_reports = self._quality.check_batch(batch)
            rejected_total += sum(1 for r in batch_reports if not r.passed)
            reports.extend(
                r for r in batch_reports if keep_passed_reports or not r.passed
            )

            chunks: list[Chunk] = []
            batch_failed: list[str] = []
            for doc_id, content in accepted:
                try:
                    chunks.extend(
                        self._chunker.chunk(
                            doc_id=doc_id, content=content, metadata=meta_by_id.get(doc_id, {})
                        )
                    )
                except Exception as exc:
                    logger.error("Chunking failed for document", doc_id=doc_id, error=str(exc))
                    batch_failed.append(doc_id)

            self._embed_and_store(chunks, batch_failed)
            failed_doc_ids.extend(d for d in batch_failed if d not in failed_doc_ids)
            accepted_total += len({d for d, _ in accepted} - set(batch_failed))
            chunk_total += len(chunks)

        result = IngestionResult(
            total_documents=len(documents),
            accepted_documents=accepted_total,
            rejected_documents=rejected_total,
            total_chunks=chunk_total,
            duration_seconds=round(time.perf_counter() - start, 2),
            failed_doc_ids=failed_doc_ids,
            quality_reports=reports,
            embed_seconds=round(self._embed_seconds, 2),
            store_seconds=round(self._store_seconds, 2),
        )
        logger.info(
            "Ingestion complete",
            total=result.total_documents,
            accepted=result.accepted_documents,
            rejected=result.rejected_documents,
            chunks=result.total_chunks,
            duration_s=result.duration_seconds,
            success_rate=result.success_rate,
        )
        return result

    def ingest_files(self, file_paths: Sequence[str | Path]) -> IngestionResult:
        """Convenience method: load text files from disk and ingest."""
        docs = []
        for path in file_paths:
            p = Path(path)
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                docs.append((p.stem, content))
            except Exception as exc:
                logger.error("Failed to read file", path=str(p), error=str(exc))

        if not docs:
            logger.warning("No documents loaded from files")
            return IngestionResult(0, 0, 0, 0, 0.0)

        return self.ingest_texts(docs)

    def _embed_and_store(
        self,
        chunks: list[Chunk],
        failed_doc_ids: list[str],
    ) -> None:
        for start in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[start : start + self._embed_batch_size]
            texts = [c.content for c in batch]

            try:
                t0 = time.perf_counter()
                embeddings = self._embedder.embed_texts(texts)
                self._embed_seconds += time.perf_counter() - t0
            except Exception as exc:
                logger.error(
                    "Embedding batch failed",
                    batch_start=start,
                    batch_size=len(batch),
                    error=str(exc),
                )
                # Mark all docs in this batch as failed rather than crashing
                for c in batch:
                    if c.doc_id not in failed_doc_ids:
                        failed_doc_ids.append(c.doc_id)
                continue

            try:
                t0 = time.perf_counter()
                self._store.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[c.metadata for c in batch],
                )
                self._store_seconds += time.perf_counter() - t0
            except Exception as exc:
                logger.error(
                    "Store upsert failed",
                    batch_start=start,
                    batch_size=len(batch),
                    error=str(exc),
                )
                for c in batch:
                    if c.doc_id not in failed_doc_ids:
                        failed_doc_ids.append(c.doc_id)
