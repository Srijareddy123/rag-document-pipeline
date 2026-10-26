"""
FastAPI application entry point.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Via Docker:
    docker compose up

Docs available at http://localhost:8000/docs after startup.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("RAG Document Pipeline starting up")
    yield
    logger.info("RAG Document Pipeline shutting down")


app = FastAPI(
    lifespan=lifespan,
    title="RAG Document Pipeline",
    description=(
        "Document ingestion and retrieval pipeline. "
        "Supports fixed-size and semantic chunking, local and Bedrock embeddings, "
        "and ChromaDB vector storage."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
