"""
Small retrieval measurement used to compare chunking strategies.

Queries are sentences taken from the documents themselves. A retrieved chunk
counts as a hit when it comes from the source document and contains at least
COVERAGE_THRESHOLD of the sentence's distinct words. Sentences cut in half by a
chunk boundary therefore do not count, which is part of what is being measured.

Two query modes: "verbatim" (the sentence as written) and "dropout" (40 percent
of the words removed at random), a rough stand-in for a vague question.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

COVERAGE_THRESHOLD = 0.8
MIN_SENTENCE_CHARS = 60
MAX_SENTENCE_CHARS = 250
MIN_DOC_CHARS = 1500
DROPOUT_RATE = 0.4
SKIP_FRACTION = 0.2  # skip the lead of each document, it often repeats the title

_WORD = re.compile(r"\w+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EvalQuery:
    doc_id: str
    sentence: str
    verbatim: str
    dropout: str


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def coverage(sentence: str, chunk: str) -> float:
    target = words(sentence)
    if not target:
        return 0.0
    return len(target & words(chunk)) / len(target)


def drop_words(sentence: str, rng: random.Random, rate: float = DROPOUT_RATE) -> str:
    tokens = sentence.split()
    kept = [t for t in tokens if rng.random() >= rate]
    return " ".join(kept) if kept else sentence


def build_queries(
    documents: Sequence[tuple[str, str]], n_queries: int, seed: int = 7
) -> list[EvalQuery]:
    """Pick one body sentence from each of n_queries randomly chosen long documents."""
    rng = random.Random(seed)
    candidates = [(d, t) for d, t in documents if len(t) >= MIN_DOC_CHARS]
    rng.shuffle(candidates)
    queries: list[EvalQuery] = []
    for doc_id, text in candidates:
        body = text[int(len(text) * SKIP_FRACTION) :]
        sentences = [
            s.strip()
            for s in _SENTENCE_SPLIT.split(body)
            if MIN_SENTENCE_CHARS <= len(s.strip()) <= MAX_SENTENCE_CHARS and "\n" not in s
        ]
        if not sentences:
            continue
        sentence = rng.choice(sentences)
        queries.append(EvalQuery(doc_id, sentence, sentence, drop_words(sentence, rng)))
        if len(queries) == n_queries:
            break
    return queries


def rank_of_evidence(hits: Sequence[dict], query: EvalQuery) -> int | None:
    """1-based rank of the first hit that is a real evidence chunk, else None."""
    for rank, hit in enumerate(hits, start=1):
        if (
            hit["id"].startswith(f"{query.doc_id}_")
            and coverage(query.sentence, hit["document"]) >= COVERAGE_THRESHOLD
        ):
            return rank
    return None


def doc_hit(hits: Sequence[dict], query: EvalQuery) -> bool:
    return any(h["id"].startswith(f"{query.doc_id}_") for h in hits)


def summarize(ranks: Sequence[int | None], doc_hits: Sequence[bool]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"queries": 0}
    return {
        "queries": n,
        "doc_hit_at_5": round(sum(doc_hits) / n, 3),
        "evidence_hit_at_1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "evidence_hit_at_5": round(sum(1 for r in ranks if r is not None) / n, 3),
        "evidence_mrr_at_5": round(sum(1 / r for r in ranks if r) / n, 3),
    }
