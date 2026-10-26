from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    documents: list[dict[str, str]] = Field(
        ...,
        description="List of documents with 'id' and 'content' keys.",
        min_length=1,
        max_length=10_000,
    )
    chunk_strategy: str = Field(
        default="fixed",
        description="Chunking strategy: fixed, semantic or recursive.",
    )
    metadata: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional per-document metadata. Must match documents length if provided.",
    )

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, docs: list[dict[str, str]]) -> list[dict[str, str]]:
        for i, doc in enumerate(docs):
            if "id" not in doc:
                raise ValueError(f"Document at index {i} missing 'id' field")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} missing 'content' field")
        return docs

    @field_validator("chunk_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("fixed", "semantic", "recursive"):
            raise ValueError(f"chunk_strategy must be fixed, semantic or recursive, got {v!r}")
        return v


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    n_results: int = Field(default=5, ge=1, le=50)
    filter_metadata: dict[str, Any] | None = Field(
        default=None,
        description="ChromaDB 'where' filter applied to metadata fields.",
    )


class QueryResult(BaseModel):
    id: str
    document: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    results: list[QueryResult]
    total_results: int
    embedding_provider: str


class IngestResponse(BaseModel):
    total_documents: int
    accepted_documents: int
    rejected_documents: int
    total_chunks: int
    duration_seconds: float
    success_rate: float
    failed_doc_ids: list[str]


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    embedding_provider: str
    chunk_strategy: str
