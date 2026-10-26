from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
)
from src.embeddings.factory import get_embedding_provider
from src.ingestion.pipeline import DocumentIngestionPipeline
from src.monitoring.logger import StructuredLogger
from src.storage.chroma import ChromaVectorStore

logger = StructuredLogger(__name__)
router = APIRouter()


def get_store() -> ChromaVectorStore:
    return ChromaVectorStore()


def get_embedder() -> Any:
    return get_embedding_provider()


@router.get("/health", response_model=HealthResponse)
def health(store: ChromaVectorStore = Depends(get_store)) -> HealthResponse:
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    strategy = os.getenv("CHUNK_STRATEGY", "fixed")
    try:
        count = store.count()
    except Exception:
        count = -1
    return HealthResponse(
        status="ok",
        vector_store_count=count,
        embedding_provider=provider,
        chunk_strategy=strategy,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    body: IngestRequest,
    store: ChromaVectorStore = Depends(get_store),
    embedder: Any = Depends(get_embedder),
) -> IngestResponse:
    pipeline = DocumentIngestionPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunk_strategy=body.chunk_strategy,
    )

    docs = [(d["id"], d["content"]) for d in body.documents]
    metadatas = body.metadata or [{} for _ in docs]

    if body.metadata and len(body.metadata) != len(docs):
        raise HTTPException(
            status_code=422,
            detail=f"metadata length ({len(body.metadata)}) must match documents length ({len(docs)})",
        )

    try:
        result = pipeline.ingest_texts(docs, metadatas=metadatas)
    except Exception as exc:
        logger.error("Ingestion endpoint error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        total_documents=result.total_documents,
        accepted_documents=result.accepted_documents,
        rejected_documents=result.rejected_documents,
        total_chunks=result.total_chunks,
        duration_seconds=result.duration_seconds,
        success_rate=result.success_rate,
        failed_doc_ids=result.failed_doc_ids,
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(
    body: QueryRequest,
    store: ChromaVectorStore = Depends(get_store),
    embedder: Any = Depends(get_embedder),
) -> QueryResponse:
    if store.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents in the vector store. Run /ingest first.",
        )

    try:
        query_embedding = embedder.embed_query(body.query)
        hits = store.query(
            query_embedding=query_embedding,
            n_results=body.n_results,
            where=body.filter_metadata,
        )
    except Exception as exc:
        logger.error("Query endpoint error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        QueryResult(
            id=h["id"],
            document=h["document"],
            score=h["score"],
            metadata=h["metadata"],
        )
        for h in hits
    ]

    return QueryResponse(
        query=body.query,
        results=results,
        total_results=len(results),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
    )
