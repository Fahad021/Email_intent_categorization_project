# tests/unit/test_telemetry.py
"""
Unit tests for classifier.telemetry

Tests cover:
- build_integrity_report()  — pure logic, no file I/O
- write_inference_record()  — writes a JSON file; validated by reading it back
"""
from __future__ import annotations

import json
import logging
import os

import pandas as pd
import pytest

from classifier.config import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SKIPPED_EMPTY,
)
from classifier.telemetry import build_integrity_report, write_inference_record

log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# build_integrity_report()
# ---------------------------------------------------------------------------

class TestIntegrityReport:

    def _make_df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"GUID": [f"g{i}" for i in range(n)]})

    def test_all_ok_passes(self):
        df = self._make_df(3)
        report = build_integrity_report(
            df_in               = df,
            predictions         = ["Billing", "Outage", "General Inquiry"],
            llm_called_flags    = [True, True, True],
            processing_statuses = [STATUS_OK, STATUS_OK, STATUS_OK],
            parquet_out         = "output/test.parquet",
            log                 = log,
        )
        assert report["passed"] is True
        assert report["row_count_match"] is True
        assert report["llm_call_match"] is True

    def test_row_count_mismatch_fails(self):
        df = self._make_df(3)
        report = build_integrity_report(
            df_in               = df,
            predictions         = ["Billing", "Outage"],   # 2 vs 3 rows
            llm_called_flags    = [True, True],
            processing_statuses = [STATUS_OK, STATUS_OK],
            parquet_out         = "output/test.parquet",
            log                 = log,
        )
        assert report["passed"] is False
        assert report["row_count_match"] is False

    def test_skipped_rows_excluded_from_llm_call_count(self):
        df = self._make_df(3)
        # row 0 skipped, rows 1+2 should call LLM
        report = build_integrity_report(
            df_in               = df,
            predictions         = ["Unclassified", "Billing", "Outage"],
            llm_called_flags    = [False, True, True],
            processing_statuses = [STATUS_SKIPPED_EMPTY, STATUS_OK, STATUS_OK],
            parquet_out         = "output/test.parquet",
            log                 = log,
        )
        assert report["passed"] is True
        assert report["rows_skipped_empty"] == 1
        assert report["expected_llm_calls"] == 2

    def test_error_rows_counted(self):
        df = self._make_df(2)
        report = build_integrity_report(
            df_in               = df,
            predictions         = ["Unclassified", "Billing"],
            llm_called_flags    = [True, True],
            processing_statuses = [STATUS_ERROR, STATUS_OK],
            parquet_out         = "output/test.parquet",
            log                 = log,
        )
        assert report["rows_errored"] == 1

    def test_empty_input_passes(self):
        df = self._make_df(0)
        report = build_integrity_report(
            df_in               = df,
            predictions         = [],
            llm_called_flags    = [],
            processing_statuses = [],
            parquet_out         = "output/test.parquet",
            log                 = log,
        )
        assert report["passed"] is True
        assert report["rows_in"] == 0


# ---------------------------------------------------------------------------
# write_inference_record()
# ---------------------------------------------------------------------------

class TestWriteInferenceRecord:

    def test_creates_json_file(self, tmp_path, base_cfg):
        write_inference_record(
            out_dir            = str(tmp_path),
            guid               = "guid-001",
            row_index          = 0,
            subject            = "Bill query",
            body               = "I was overcharged",
            user_prompt        = "Subject: Bill query\nBody:\nI was overcharged",
            system_prompt      = "You are a classifier.",
            raw_response       = '{"intent_category": "Billing"}',
            label              = "Billing",
            parse_status       = STATUS_OK,
            processing_status  = STATUS_OK,
            llm_called         = True,
            latency_ms         = 250,
            parquet_in         = "data/emails.parquet",
            prompt_tokens      = 100,
            completion_tokens  = 20,
            total_tokens       = 120,
            confidence         = "high",
            all_intents        = ["Billing"],
            body_budget        = 2000,
            cfg                = base_cfg,
        )
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_json_content_is_valid(self, tmp_path, base_cfg):
        write_inference_record(
            out_dir            = str(tmp_path),
            guid               = "guid-002",
            row_index          = 1,
            subject            = "No power",
            body               = "Power outage since 8am",
            user_prompt        = "Subject: No power\nBody:\nPower outage since 8am",
            system_prompt      = "You are a classifier.",
            raw_response       = '{"intent_category": "Outage"}',
            label              = "Outage",
            parse_status       = STATUS_OK,
            processing_status  = STATUS_OK,
            llm_called         = True,
            latency_ms         = 180,
            parquet_in         = "data/emails.parquet",
            prompt_tokens      = 90,
            completion_tokens  = 15,
            total_tokens       = 105,
            confidence         = "high",
            all_intents        = ["Outage"],
            body_budget        = 2000,
            cfg                = base_cfg,
        )
        path = next(tmp_path.glob("*.json"))
        with open(path, encoding="utf-8") as f:
            record = json.load(f)

        assert record["prediction"]["intent_code"] == "Outage"
        assert record["prediction"]["llm_called"] is True
        assert record["prediction"]["confidence"] == "high"
        assert record["timing"]["latency_ms"] == 180
