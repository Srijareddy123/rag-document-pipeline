"""
Chunking strategies.

Three strategies:

- fixed: character windows with overlap. Predictable chunk count.
- semantic: splits on paragraph and sentence boundaries, then merges small
  pieces up to a size limit. It does not use embeddings, so "semantic" here
  means "follows the structure of the text".
- recursive: LangChain's RecursiveCharacterTextSplitter, kept as a reference
  point for the chunking comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.monitoring.logger import StructuredLogger

logger = StructuredLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    total_chunks: int
    strategy: str
    metadata: dict


class FixedSizeChunker:
    """
    Splits documents into fixed-character windows with overlap.

    chunk_size=1000 with overlap=200 gives ~80% unique content per chunk,
    which is a reasonable tradeoff for general-purpose retrieval.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc_id: str, content: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        step = self.chunk_size - self.overlap
        positions = list(range(0, len(content), step))

        # Remove trailing position that would produce an empty chunk
        positions = [p for p in positions if p < len(content)]

        total = len(positions)
        for idx, start in enumerate(positions):
            end = min(start + self.chunk_size, len(content))
            text = content[start:end].strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{idx:04d}",
                    doc_id=doc_id,
                    content=text,
                    chunk_index=idx,
                    total_chunks=total,
                    strategy="fixed_size",
                    metadata={**metadata, "start_char": start, "end_char": end},
                )
            )

        logger.debug(
            "Fixed-size chunking complete",
            doc_id=doc_id,
            total_chunks=len(chunks),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        return chunks


class SemanticChunker:
    """
    Splits on paragraph or sentence boundaries, then merges small fragments
    up to max_chunk_size to avoid sending single-sentence embeddings.

    This produces variable-length chunks that tend to align with the actual
    semantic units in the document. Trade-off: less predictable chunk count,
    slightly more preprocessing time.
    """

    def __init__(self, max_chunk_size: int = 1200, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, doc_id: str, content: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        paragraphs = self._split_paragraphs(content)
        merged = self._merge_fragments(paragraphs)

        chunks = []
        for idx, text in enumerate(merged):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_sem_{idx:04d}",
                    doc_id=doc_id,
                    content=text.strip(),
                    chunk_index=idx,
                    total_chunks=len(merged),
                    strategy="semantic",
                    metadata={**metadata, "paragraph_index": idx},
                )
            )

        logger.debug(
            "Semantic chunking complete",
            doc_id=doc_id,
            total_chunks=len(chunks),
        )
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        # Split on double newlines (paragraph boundaries) first,
        # then fall back to sentence boundaries for very long paragraphs
        raw_blocks = re.split(r"\n{2,}", text)
        result = []
        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue
            if len(block) <= self.max_chunk_size:
                result.append(block)
            else:
                # Long paragraph, split on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", block)
                result.extend(s for s in sentences if s.strip())
        return result

    def _merge_fragments(self, fragments: list[str]) -> list[str]:
        """Greedily merge adjacent fragments until max_chunk_size."""
        merged = []
        current = ""
        for frag in fragments:
            if not current:
                current = frag
            elif len(current) + len(frag) + 1 <= self.max_chunk_size:
                current += "\n" + frag
            else:
                if len(current) >= self.min_chunk_size:
                    merged.append(current)
                elif merged:
                    # Attach tiny fragment to previous chunk rather than orphaning it
                    merged[-1] += "\n" + current
                else:
                    merged.append(current)
                current = frag

        if current:
            if len(current) >= self.min_chunk_size:
                merged.append(current)
            elif merged:
                merged[-1] += "\n" + current
            else:
                merged.append(current)

        return merged


class RecursiveChunker:
    """LangChain RecursiveCharacterTextSplitter behind the same interface."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )

    def chunk(self, doc_id: str, content: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        pieces = [p.strip() for p in self._splitter.split_text(content) if p.strip()]
        return [
            Chunk(
                chunk_id=f"{doc_id}_rec_{idx:04d}",
                doc_id=doc_id,
                content=text,
                chunk_index=idx,
                total_chunks=len(pieces),
                strategy="recursive",
                metadata={**metadata},
            )
            for idx, text in enumerate(pieces)
        ]


STRATEGIES = ("fixed", "semantic", "recursive")


def get_chunker(strategy: str, **kwargs) -> FixedSizeChunker | SemanticChunker | RecursiveChunker:
    if strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    raise ValueError(
        f"Unknown chunking strategy: {strategy!r}. Use one of {', '.join(STRATEGIES)}."
    )
