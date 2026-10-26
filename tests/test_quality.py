"""Tests for document quality checks."""

from src.ingestion.quality import DocumentQualityChecker

VALID_DOC = (
    "This is a valid document with enough content to pass quality checks. "
    "It contains readable text with a good ratio of alphabetic characters. " * 3
)


class TestDocumentQualityChecker:
    def setup_method(self):
        self.checker = DocumentQualityChecker()

    def test_valid_document_passes(self):
        report = self.checker.check("doc1", VALID_DOC)
        assert report.passed is True

    def test_empty_document_fails(self):
        report = self.checker.check("doc1", "")
        assert report.passed is False
        assert "not_empty" in report.reason

    def test_whitespace_only_fails(self):
        report = self.checker.check("doc1", "   \n\t  ")
        assert report.passed is False

    def test_too_short_fails(self):
        report = self.checker.check("doc1", "Hi.")
        assert report.passed is False
        assert "not_empty" in report.reason

    def test_mostly_symbols_fails(self):
        report = self.checker.check("doc1", "!@#$%^&*()!@#$%^" * 20)
        assert report.passed is False
        assert "has_readable_text" in report.reason

    def test_null_bytes_fail(self):
        report = self.checker.check("doc1", VALID_DOC + "\x00\x00")
        assert report.passed is False
        assert "no_null_bytes" in report.reason

    def test_replacement_chars_fail(self):
        report = self.checker.check("doc1", VALID_DOC + "�" * 5)
        assert report.passed is False
        assert "no_replacement_chars" in report.reason

    def test_exact_duplicate_fails(self):
        self.checker.check("doc1", VALID_DOC)
        report = self.checker.check("doc2", VALID_DOC)
        assert report.passed is False
        assert "not_duplicate" in report.reason

    def test_whitespace_normalized_for_dedup(self):
        """Extra whitespace should not prevent dedup detection."""
        doc_a = "Hello world this is a test document with enough content." * 5
        doc_b = "Hello  world   this is  a test document with enough content." * 5
        self.checker.check("doc1", doc_a)
        report = self.checker.check("doc2", doc_b)
        assert report.passed is False

    def test_different_documents_both_pass(self):
        doc_a = VALID_DOC
        doc_b = (
            "Entirely different content about vector databases and embeddings. "
            "This document talks about HNSW indexes and cosine similarity. " * 3
        )
        r1 = self.checker.check("doc1", doc_a)
        r2 = self.checker.check("doc2", doc_b)
        assert r1.passed is True
        assert r2.passed is True

    def test_report_contains_doc_id(self):
        report = self.checker.check("my_special_id", VALID_DOC)
        assert report.doc_id == "my_special_id"

    def test_report_repr_includes_status(self):
        report = self.checker.check("doc1", VALID_DOC)
        assert "PASS" in repr(report)

    def test_check_batch_returns_accepted_and_reports(self):
        docs = [
            ("good1", VALID_DOC),
            ("bad1", ""),
            ("good2", VALID_DOC[::-1][:200] + VALID_DOC),  # different content
        ]
        # Use fresh checker to avoid cross-test dedup state
        checker = DocumentQualityChecker()
        accepted, reports = checker.check_batch(docs)
        assert len(reports) == 3
        # good1 and good2 should pass; bad1 should fail
        passed_ids = {r.doc_id for r in reports if r.passed}
        assert "good1" in passed_ids
        assert "bad1" not in passed_ids

    def test_batch_accepted_subset_matches_reports(self):
        checker = DocumentQualityChecker()
        docs = [("a", VALID_DOC), ("b", "too short")]
        accepted, reports = checker.check_batch(docs)
        accepted_ids = {doc_id for doc_id, _ in accepted}
        passed_report_ids = {r.doc_id for r in reports if r.passed}
        assert accepted_ids == passed_report_ids

    def test_seen_fingerprints_can_be_seeded(self):
        """Pre-seeding fingerprints lets you deduplicate across pipeline runs."""
        import hashlib
        import re

        from src.ingestion.quality import DocumentQualityChecker

        content = VALID_DOC
        normalized = re.sub(r"\s+", " ", content.strip().lower())
        fp = hashlib.sha256(normalized.encode()).hexdigest()

        checker = DocumentQualityChecker(seen_fingerprints={fp})
        report = checker.check("doc1", content)
        assert report.passed is False
        assert "not_duplicate" in report.reason
