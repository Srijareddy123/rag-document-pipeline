"""
Data quality checks run before a document enters the ingestion pipeline.

These exist because real document dumps are messy: encoding artifacts,
near-empty files, duplicates, and token-heavy binary payloads that bloat
embeddings without adding retrieval value.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.monitoring.logger import StructuredLogger

logger = StructuredLogger(__name__)

MIN_CONTENT_LENGTH = 50
MAX_CONTENT_LENGTH = 2_000_000  # ~2MB of raw text
MIN_ALPHA_RATIO = 0.4           # reject docs that are mostly symbols/numbers
MAX_DUPLICATE_RATIO = 0.85      # reject near-duplicate blocks


@dataclass
class QualityReport:
    doc_id: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reason: str = ""

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"QualityReport({self.doc_id}, {status}: {self.reason or 'ok'})"


class DocumentQualityChecker:
    """
    Runs a battery of lightweight checks before chunking.

    Designed to be fast (no ML inference here), each check is O(n) in
    document length at worst. The dedup check is O(1) per doc with a
    rolling fingerprint set.
    """

    def __init__(self, seen_fingerprints: set[str] | None = None):
        self._seen: set[str] = seen_fingerprints or set()

    def check(self, doc_id: str, content: str) -> QualityReport:
        checks: dict[str, bool] = {}

        # --- length ---
        checks["not_empty"] = len(content.strip()) >= MIN_CONTENT_LENGTH
        checks["not_oversized"] = len(content) <= MAX_CONTENT_LENGTH

        # --- character composition ---
        alpha_count = sum(1 for c in content if c.isalpha())
        ratio = alpha_count / max(len(content), 1)
        checks["has_readable_text"] = ratio >= MIN_ALPHA_RATIO

        # --- encoding artifacts ---
        checks["no_null_bytes"] = "\x00" not in content
        checks["no_replacement_chars"] = "�" not in content

        # --- exact duplicate ---
        fingerprint = self._fingerprint(content)
        is_dup = fingerprint in self._seen
        checks["not_duplicate"] = not is_dup
        if not is_dup:
            self._seen.add(fingerprint)

        passed = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        reason = ", ".join(failed) if failed else ""

        if not passed:
            logger.warning(
                "Document failed quality check",
                doc_id=doc_id,
                failed_checks=failed,
            )

        return QualityReport(doc_id=doc_id, passed=passed, checks=checks, reason=reason)

    def check_batch(
        self, docs: Sequence[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], list[QualityReport]]:
        """
        Returns (accepted_docs, all_reports).
        accepted_docs keeps (doc_id, content) pairs that passed all checks.
        """
        accepted = []
        reports = []
        for doc_id, content in docs:
            report = self.check(doc_id, content)
            reports.append(report)
            if report.passed:
                accepted.append((doc_id, content))
        return accepted, reports

    @staticmethod
    def _fingerprint(content: str) -> str:
        normalized = re.sub(r"\s+", " ", content.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()
