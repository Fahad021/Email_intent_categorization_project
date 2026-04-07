# tests/unit/test_inference.py
"""
Unit tests for classifier.inference.parse_llm_output()

All tests are pure Python — no model, no file I/O, no network.
"""
from __future__ import annotations

import pytest

from classifier.config import (
    STATUS_BAD_JSON,
    STATUS_CASE_CORRECTED,
    STATUS_EMPTY_FIELDS,
    STATUS_INVALID_CATEGORY,
    STATUS_NO_JSON,
    STATUS_OK,
)
from classifier.inference import parse_llm_output

LABELS = ["Billing", "Outage", "New Connection", "General Inquiry"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(raw: str):
    return parse_llm_output(raw, LABELS)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_exact_label_match(self):
        raw = '{"intent_category": "Billing", "confidence": "high", "all_intents": ["Billing"]}'
        label, _, status, confidence, all_intents = parse(raw)
        assert label == "Billing"
        assert status == STATUS_OK
        assert confidence == "high"
        assert all_intents == ["Billing"]

    def test_intent_code_field_used_as_fallback(self):
        raw = '{"intent_code": "Outage", "confidence": "medium", "all_intents": []}'
        label, _, status, _, _ = parse(raw)
        assert label == "Outage"
        assert status == STATUS_OK

    def test_markdown_fenced_json(self):
        raw = '```json\n{"intent_category": "Billing", "confidence": "low", "all_intents": []}\n```'
        label, _, status, _, _ = parse(raw)
        assert label == "Billing"
        assert status == STATUS_OK

    def test_trailing_comma_tolerated(self):
        raw = '{"intent_category": "Outage", "confidence": "high", "all_intents": [],}'
        label, _, status, _, _ = parse(raw)
        assert label == "Outage"
        assert status == STATUS_OK

    def test_all_intents_multi_value(self):
        raw = '{"intent_category": "Billing", "confidence": "high", "all_intents": ["Billing", "Outage"]}'
        _, _, _, _, all_intents = parse(raw)
        assert "Billing" in all_intents
        assert "Outage" in all_intents


# ---------------------------------------------------------------------------
# Case correction
# ---------------------------------------------------------------------------

class TestCaseCorrection:

    def test_lowercase_label_corrected(self):
        raw = '{"intent_category": "billing", "confidence": "high", "all_intents": []}'
        label, _, status, _, _ = parse(raw)
        assert label == "Billing"
        assert status == STATUS_CASE_CORRECTED

    def test_uppercase_label_corrected(self):
        raw = '{"intent_category": "OUTAGE", "confidence": "high", "all_intents": []}'
        label, _, status, _, _ = parse(raw)
        assert label == "Outage"
        assert status == STATUS_CASE_CORRECTED


# ---------------------------------------------------------------------------
# Confidence normalisation
# ---------------------------------------------------------------------------

class TestConfidence:

    @pytest.mark.parametrize("conf", ["high", "medium", "low"])
    def test_valid_confidence_preserved(self, conf):
        raw = f'{{"intent_category": "Billing", "confidence": "{conf}", "all_intents": []}}'
        _, _, _, confidence, _ = parse(raw)
        assert confidence == conf

    def test_invalid_confidence_becomes_unknown(self):
        raw = '{"intent_category": "Billing", "confidence": "very sure", "all_intents": []}'
        _, _, _, confidence, _ = parse(raw)
        assert confidence == "unknown"

    def test_missing_confidence_becomes_unknown(self):
        raw = '{"intent_category": "Billing", "all_intents": []}'
        _, _, _, confidence, _ = parse(raw)
        assert confidence == "unknown"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

class TestNoJson:

    def test_plain_text_returns_no_json(self):
        label, _, status, _, _ = parse("I think it is billing related")
        assert label == "Unclassified"
        assert status == STATUS_NO_JSON

    def test_empty_string(self):
        label, _, status, _, _ = parse("")
        assert label == "Unclassified"
        assert status == STATUS_NO_JSON

    def test_none_input(self):
        label, _, status, _, _ = parse(None)  # type: ignore[arg-type]
        assert label == "Unclassified"
        assert status == STATUS_NO_JSON


class TestBadJson:

    def test_malformed_json(self):
        raw = '{"intent_category": "Billing" "confidence": "high"}'  # missing comma
        label, _, status, _, _ = parse(raw)
        assert label == "Unclassified"
        assert status == STATUS_BAD_JSON


class TestEmptyFields:

    def test_both_fields_blank(self):
        raw = '{"intent_category": "", "intent_code": "", "confidence": "high", "all_intents": []}'
        label, _, status, _, _ = parse(raw)
        assert label == "Unclassified"
        assert status == STATUS_EMPTY_FIELDS


class TestInvalidCategory:

    def test_unknown_label(self):
        raw = '{"intent_category": "Complaints", "confidence": "high", "all_intents": []}'
        label, _, status, _, _ = parse(raw)
        assert label == "Unclassified"
        assert status == STATUS_INVALID_CATEGORY

    def test_numeric_label(self):
        raw = '{"intent_category": "12345", "confidence": "high", "all_intents": []}'
        label, _, status, _, _ = parse(raw)
        assert label == "Unclassified"
        assert status == STATUS_INVALID_CATEGORY
