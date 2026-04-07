# tests/integration/test_parquet_telemetry.py
"""
Integration test: full parquet write → telemetry → integrity check cycle.

Simulates what the pipeline does after inference:
  1. Write output parquet with predictions
  2. Write per-row inference records (JSON files)
  3. Call build_integrity_report() and verify the report
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from classifier.config import STATUS_OK, STATUS_SKIPPED_EMPTY
from classifier.data_io import read_parquet, write_parquet
from classifier.telemetry import build_integrity_report, write_inference_record
import logging

log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_records(tmp_path, cfg, rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        write_inference_record(
            out_dir            = str(tmp_path / "records"),
            guid               = row["guid"],
            row_index          = i,
            subject            = row["subject"],
            body               = row["body"],
            user_prompt        = f"Subject: {row['subject']}\nBody:\n{row['body']}",
            system_prompt      = "You are a classifier.",
            raw_response       = row["raw"],
            label              = row["label"],
            parse_status       = row["status"],
            processing_status  = row["status"],
            llm_called         = row.get("llm_called", True),
            latency_ms         = 100,
            parquet_in         = "data/emails.parquet",
            prompt_tokens      = 50,
            completion_tokens  = 10,
            total_tokens       = 60,
            confidence         = "high",
            all_intents        = [row["label"]],
            body_budget        = 2000,
            cfg                = cfg,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParquetTelemetryIntegration:

    ROWS = [
        {"guid": "g1", "subject": "Bill",   "body": "overcharged",  "label": "Billing",       "raw": '{"intent_category": "Billing"}',       "status": STATUS_OK,           "llm_called": True},
        {"guid": "g2", "subject": "Outage", "body": "no power",     "label": "Outage",         "raw": '{"intent_category": "Outage"}',         "status": STATUS_OK,           "llm_called": True},
        {"guid": "g3", "subject": "",       "body": "",             "label": "Unclassified",   "raw": "",                                     "status": STATUS_SKIPPED_EMPTY, "llm_called": False},
    ]

    def test_output_parquet_round_trip(self, tmp_path, base_cfg):
        df = pd.DataFrame({
            "GUID":    [r["guid"]    for r in self.ROWS],
            "Subject": [r["subject"] for r in self.ROWS],
            "Body":    [r["body"]    for r in self.ROWS],
            "Predicted_Reduced_Category": [r["label"] for r in self.ROWS],
        })
        out_path = str(tmp_path / "output.parquet")
        write_parquet(df, out_path)
        df_back = read_parquet(out_path)
        assert list(df_back["Predicted_Reduced_Category"]) == [r["label"] for r in self.ROWS]

    def test_inference_records_written(self, tmp_path, base_cfg):
        _write_records(tmp_path, base_cfg, self.ROWS)
        files = list((tmp_path / "records").glob("*.json"))
        assert len(files) == len(self.ROWS)

    def test_inference_records_valid_json(self, tmp_path, base_cfg):
        _write_records(tmp_path, base_cfg, self.ROWS)
        for f in (tmp_path / "records").glob("*.json"):
            with open(f, encoding="utf-8") as fh:
                record = json.load(fh)
            assert "prediction" in record
            assert "timing" in record

    def test_integrity_report_passes(self, tmp_path, base_cfg):
        df_in = pd.DataFrame({"GUID": [r["guid"] for r in self.ROWS]})
        report = build_integrity_report(
            df_in               = df_in,
            predictions         = [r["label"]      for r in self.ROWS],
            llm_called_flags    = [r["llm_called"]  for r in self.ROWS],
            processing_statuses = [r["status"]      for r in self.ROWS],
            parquet_out         = str(tmp_path / "output.parquet"),
            log                 = log,
        )
        assert report["passed"] is True
        assert report["rows_in"]  == 3
        assert report["rows_out"] == 3
        assert report["rows_skipped_empty"] == 1
        assert report["rows_llm_called"]    == 2

    def test_integrity_report_embedded_in_manifest(self, tmp_path, base_cfg):
        """Verify the report dict structure is serialisable (fits in manifest)."""
        df_in = pd.DataFrame({"GUID": [r["guid"] for r in self.ROWS]})
        report = build_integrity_report(
            df_in               = df_in,
            predictions         = [r["label"]      for r in self.ROWS],
            llm_called_flags    = [r["llm_called"]  for r in self.ROWS],
            processing_statuses = [r["status"]      for r in self.ROWS],
            parquet_out         = str(tmp_path / "output.parquet"),
            log                 = log,
        )
        # Must be JSON-serialisable
        serialised = json.dumps(report)
        parsed     = json.loads(serialised)
        assert parsed["passed"] is True
