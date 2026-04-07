# tests/integration/test_pipeline_run.py
"""
Integration test for classifier.pipeline.run()

The LLM model is fully stubbed — no GGUF file needed.
Tests confirm the end-to-end pipeline writes a valid output parquet
and a manifest file when given a small in-memory email dataset.
"""
from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from classifier.config import (
    Config,
    STATUS_OK,
    STATUS_SKIPPED_EMPTY,
)
from classifier.pipeline import run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_kb(tmp_path) -> str:
    """Write a tiny KB Excel file."""
    path = str(tmp_path / "kb.xlsx")
    df = pd.DataFrame({
        "Reduced_Category": ["Billing", "Outage", "General Inquiry"],
        "Merged_Terms":     ["bill, invoice, charge", "power, blackout", "help, question"],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Merged_Knowledgebase", index=False)
    return path


def _make_parquet(tmp_path) -> str:
    """Write a 4-row email parquet (row 3 is empty to test skipping)."""
    path = str(tmp_path / "emails.parquet")
    df = pd.DataFrame({
        "GUID":    ["g1", "g2", "g3", "g4"],
        "Subject": ["My bill is wrong", "Power is out", "Need info", ""],
        "Body":    ["I was overcharged", "No electricity", "How do I sign up?", ""],
    })
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def _make_cfg(tmp_path, parquet_in: str, kb_file: str) -> Config:
    return Config(
        model_path  = str(tmp_path / "model.gguf"),
        kb_file     = kb_file,
        parquet_in  = parquet_in,
        parquet_out = str(tmp_path / "output.parquet"),
        data_source = "parquet",
        run_mode    = "test",
        no_records  = True,          # skip JSON telemetry files
        json_logs   = False,
        log_level   = "WARNING",     # quiet
        record_dir  = str(tmp_path / "records"),
    )


# ---------------------------------------------------------------------------
# Mock LLM response factory
# ---------------------------------------------------------------------------

def _llm_response(label: str) -> dict:
    return {
        "choices": [{
            "text": (
                f'{{"intent_category": "{label}", '
                f'"confidence": "high", '
                f'"all_intents": ["{label}"]}}'
            )
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineRun:

    def _run_with_mock(self, cfg: Config, labels: list[str]):
        """
        Patch load_model, the Llama __call__, and tokenize so no real model is needed.
        Cycles through 'labels' for successive non-empty rows.
        """
        call_count = 0
        mock_model = MagicMock()
        mock_model.tokenize.side_effect = lambda b: list(range(max(1, len(b) // 4)))

        def fake_call(**kwargs):
            nonlocal call_count
            label = labels[call_count % len(labels)]
            call_count += 1
            return _llm_response(label)

        mock_model.side_effect = None
        mock_model.__call__ = MagicMock(side_effect=fake_call)
        mock_model.return_value = _llm_response(labels[0])

        # Make __call__ work both as mock_model(...) and mock_model.__call__(...)
        type(mock_model).__call__ = lambda self, **kw: fake_call(**kw)

        with patch("classifier.pipeline.load_model", return_value=mock_model), \
             patch("classifier.pipeline.load_reduced_kb") as mock_kb, \
             patch("classifier.inference.Llama"):
            mock_kb.return_value = (
                ["Billing", "Outage", "General Inquiry"],
                {"Billing": ["bill"], "Outage": ["power"], "General Inquiry": ["help"]},
            )
            # Also patch predict_intent directly so we don't need a real LLM call
            with patch("classifier.pipeline.predict_intent") as mock_pi:
                side_effects = []
                for lbl in labels:
                    side_effects.append((
                        lbl,
                        f'{{"intent_category": "{lbl}"}}',
                        STATUS_OK,
                        {
                            "system_prompt":     "sys",
                            "user_prompt":       "usr",
                            "prompt_tokens":     50,
                            "completion_tokens": 10,
                            "total_tokens":      60,
                            "confidence":        "high",
                            "all_intents":       [lbl],
                        },
                    ))
                mock_pi.side_effect = side_effects
                run(cfg)

    def test_output_parquet_created(self, tmp_path):
        parquet_in = _make_parquet(tmp_path)
        kb_file    = _make_kb(tmp_path)
        cfg        = _make_cfg(tmp_path, parquet_in, kb_file)

        self._run_with_mock(cfg, ["Billing", "Outage", "General Inquiry"])

        assert os.path.isfile(cfg.parquet_out)

    def test_output_row_count_matches_input(self, tmp_path):
        parquet_in = _make_parquet(tmp_path)
        kb_file    = _make_kb(tmp_path)
        cfg        = _make_cfg(tmp_path, parquet_in, kb_file)

        self._run_with_mock(cfg, ["Billing", "Outage", "General Inquiry"])

        df_out = pd.read_parquet(cfg.parquet_out)
        df_in  = pd.read_parquet(parquet_in)
        assert len(df_out) == len(df_in)

    def test_output_contains_prediction_column(self, tmp_path):
        parquet_in = _make_parquet(tmp_path)
        kb_file    = _make_kb(tmp_path)
        cfg        = _make_cfg(tmp_path, parquet_in, kb_file)

        self._run_with_mock(cfg, ["Billing", "Outage", "General Inquiry"])

        df_out = pd.read_parquet(cfg.parquet_out)
        assert cfg.out_col in df_out.columns

    def test_empty_row_is_unclassified(self, tmp_path):
        parquet_in = _make_parquet(tmp_path)   # row 3 is empty
        kb_file    = _make_kb(tmp_path)
        cfg        = _make_cfg(tmp_path, parquet_in, kb_file)

        self._run_with_mock(cfg, ["Billing", "Outage", "General Inquiry"])

        df_out = pd.read_parquet(cfg.parquet_out)
        # Last row had empty subject+body — must be Unclassified
        assert df_out.iloc[-1][cfg.out_col] == "Unclassified"

    def test_manifest_file_created(self, tmp_path):
        parquet_in = _make_parquet(tmp_path)
        kb_file    = _make_kb(tmp_path)
        cfg        = _make_cfg(tmp_path, parquet_in, kb_file)

        self._run_with_mock(cfg, ["Billing", "Outage", "General Inquiry"])

        manifest_path = cfg.parquet_out + ".manifest.json"
        assert os.path.isfile(manifest_path)

    def test_manifest_is_valid_json(self, tmp_path):
        parquet_in = _make_parquet(tmp_path)
        kb_file    = _make_kb(tmp_path)
        cfg        = _make_cfg(tmp_path, parquet_in, kb_file)

        self._run_with_mock(cfg, ["Billing", "Outage", "General Inquiry"])

        manifest_path = cfg.parquet_out + ".manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["rows"] == 4
        assert "integrity" in manifest
