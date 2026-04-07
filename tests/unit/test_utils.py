# tests/unit/test_utils.py
"""
Unit tests for classifier.utils
- redact()      PII patterns
- sha256_str()  determinism + empty string
- preview()     truncation
- ensure_dir()  idempotency
"""
from __future__ import annotations

import os

import pytest

from classifier.utils import ensure_dir, preview, redact, sha256_str


# ---------------------------------------------------------------------------
# redact()
# ---------------------------------------------------------------------------

class TestRedact:

    def test_email_redacted(self):
        result = redact("Contact person@example.com for help")
        assert "<REDACTED_EMAIL>" in result
        assert "person@example.com" not in result

    def test_phone_redacted(self):
        result = redact("Call us at 416-555-1234 today")
        assert "<REDACTED_PHONE>" in result
        assert "416-555-1234" not in result

    def test_long_account_number_redacted(self):
        # 10-digit number triggers <REDACTED_NUMBER>
        result = redact("Account 1234567890 is overdue")
        assert "<REDACTED_NUMBER>" in result

    def test_short_number_not_redacted(self):
        # Numbers shorter than 9 digits should NOT be redacted
        result = redact("I have 3 invoices")
        assert "3" in result

    def test_redact_disabled(self):
        text = "email me at test@test.com"
        result = redact(text, enabled=False)
        assert "test@test.com" in result

    def test_truncates_to_max_chars(self):
        long_text = "a" * 500
        result = redact(long_text, max_chars=100)
        assert len(result) <= 100

    def test_none_input(self):
        assert redact(None) == ""  # type: ignore[arg-type]

    def test_empty_string(self):
        assert redact("") == ""

    def test_multiple_emails_redacted(self):
        result = redact("From: a@b.com To: c@d.com")
        assert result.count("<REDACTED_EMAIL>") == 2


# ---------------------------------------------------------------------------
# sha256_str()
# ---------------------------------------------------------------------------

class TestSha256Str:

    def test_deterministic(self):
        assert sha256_str("hello") == sha256_str("hello")

    def test_different_inputs_differ(self):
        assert sha256_str("hello") != sha256_str("world")

    def test_empty_string(self):
        h = sha256_str("")
        assert isinstance(h, str) and len(h) == 64

    def test_none_input(self):
        h = sha256_str(None)  # type: ignore[arg-type]
        assert isinstance(h, str) and len(h) == 64

    def test_returns_hex(self):
        h = sha256_str("test")
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# preview()
# ---------------------------------------------------------------------------

class TestPreview:

    def test_short_string_unchanged(self):
        assert preview("hello", max_chars=100) == "hello"

    def test_truncates_at_max_chars(self):
        assert preview("abcde", max_chars=3) == "abc"

    def test_none_returns_empty(self):
        assert preview(None) == ""  # type: ignore[arg-type]

    def test_default_max_chars(self):
        long = "x" * 400
        assert len(preview(long)) == 300


# ---------------------------------------------------------------------------
# ensure_dir()
# ---------------------------------------------------------------------------

class TestEnsureDir:

    def test_creates_directory(self, tmp_path):
        target = str(tmp_path / "new_dir" / "sub")
        ensure_dir(target)
        assert os.path.isdir(target)

    def test_idempotent(self, tmp_path):
        target = str(tmp_path / "dir")
        ensure_dir(target)
        ensure_dir(target)   # should not raise
        assert os.path.isdir(target)
