"""
ChromaDB vector store wrapper.

Runs in-process with a persistent directory by default, or against a ChromaDB
server when CHROMA_HOST is set (docker compose does this).
"""

from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from src.monitoring.logger import StructuredLogger  # noqa: E402

logger = StructuredLogger(__name__)


class ChromaVectorStore:
    """
    Manages a ChromaDB collection: upsert, query, delete.

    One collection holds one logical document set. Cosine distance.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        self._collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", "documents"
        )
        self._persist_dir = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./chroma_data"
        )
        self._host = host or os.getenv("CHROMA_HOST", "")
        self._port = port or int(os.getenv("CHROMA_PORT", "8001"))
        self._client = None
        self._collection = None

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is not None:
            return self._client

        if self._host:
            logger.info(
                "Connecting to remote ChromaDB",
                host=self._host,
                port=self._port,
            )
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
            )
        else:
            logger.info(
                "Using local persistent ChromaDB",
                persist_dir=self._persist_dir,
            )
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is not None:
            return self._collection
        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection ready",
            collection=self._collection_name,
            count=self._collection.count(),
        )
        return self._collection

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        if not ids:
            return
        # ChromaDB 0.5+ rejects empty dicts in metadata, use None for empty entries
        if metadatas is None:
            metadatas = [None for _ in ids]
        else:
            metadatas = [m if m else None for m in metadatas]

        col = self._get_collection()

        # ChromaDB has an undocumented batch size limit around 5461 items
        # (related to SQLite's max parameter count). Chunk the upsert to be safe.
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            col.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            logger.debug(
                "Upserted batch to ChromaDB",
                batch_start=start,
                batch_end=min(end, len(ids)),
                collection=self._collection_name,
            )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        col = self._get_collection()
        count = col.count()
        if count == 0:
            return []

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] or {},
                    "distance": results["distances"][0][i],
                    "score": round(1.0 - results["distances"][0][i], 4),
                }
            )
        return hits

    def delete_collection(self) -> None:
        client = self._get_client()
        client.delete_collection(self._collection_name)
        self._collection = None
        logger.info("Collection deleted", collection=self._collection_name)

    def count(self) -> int:
        return self._get_collection().count()
