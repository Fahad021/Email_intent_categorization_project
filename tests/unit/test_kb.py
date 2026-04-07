# tests/unit/test_kb.py
"""
Unit tests for classifier.kb.load_reduced_kb

Uses an in-memory Excel fixture written to a temp file — no shared fixture files needed.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from classifier.kb import load_reduced_kb

log = logging.getLogger("test")


def _write_kb(tmp_path, rows: list[dict]) -> str:
    """Write a minimal KB Excel file and return its path."""
    path = str(tmp_path / "kb.xlsx")
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Merged_Knowledgebase", index=False)
    return path


class TestLoadReducedKb:

    def test_returns_labels_and_terms(self, tmp_path):
        path = _write_kb(tmp_path, [
            {"Reduced_Category": "Billing",  "Merged_Terms": "invoice, bill, charge"},
            {"Reduced_Category": "Outage",   "Merged_Terms": "power, electricity, blackout"},
        ])
        labels, terms = load_reduced_kb(path, log)
        assert "Billing" in labels
        assert "Outage" in labels

    def test_labels_are_sorted(self, tmp_path):
        path = _write_kb(tmp_path, [
            {"Reduced_Category": "Outage",  "Merged_Terms": "power"},
            {"Reduced_Category": "Billing", "Merged_Terms": "bill"},
        ])
        labels, _ = load_reduced_kb(path, log)
        assert labels == sorted(labels)

    def test_terms_are_split_by_comma(self, tmp_path):
        path = _write_kb(tmp_path, [
            {"Reduced_Category": "Billing", "Merged_Terms": "invoice, bill, charge"},
        ])
        _, terms = load_reduced_kb(path, log)
        assert "invoice" in terms["Billing"]
        assert "bill" in terms["Billing"]
        assert "charge" in terms["Billing"]

    def test_terms_lowercased(self, tmp_path):
        path = _write_kb(tmp_path, [
            {"Reduced_Category": "Billing", "Merged_Terms": "Invoice, BILL"},
        ])
        _, terms = load_reduced_kb(path, log)
        assert "invoice" in terms["Billing"]
        assert "bill" in terms["Billing"]

    def test_multiple_rows_same_category_merged(self, tmp_path):
        path = _write_kb(tmp_path, [
            {"Reduced_Category": "Billing", "Merged_Terms": "invoice"},
            {"Reduced_Category": "Billing", "Merged_Terms": "bill"},
        ])
        _, terms = load_reduced_kb(path, log)
        assert "invoice" in terms["Billing"]
        assert "bill" in terms["Billing"]

    def test_nan_rows_dropped(self, tmp_path):
        path = _write_kb(tmp_path, [
            {"Reduced_Category": "Billing", "Merged_Terms": "invoice"},
            {"Reduced_Category": None,      "Merged_Terms": None},
        ])
        labels, _ = load_reduced_kb(path, log)
        assert None not in labels
        assert len(labels) == 1
