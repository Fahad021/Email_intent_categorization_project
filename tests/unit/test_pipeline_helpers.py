# tests/unit/test_pipeline_helpers.py
"""
Unit tests for the pure helper functions in classifier.pipeline:
- _resolve_output_path()
- _classify_row()       with predict_intent mocked
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from classifier.config import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SKIPPED_EMPTY,
)
from classifier.pipeline import _classify_row, _resolve_output_path


# ---------------------------------------------------------------------------
# _resolve_output_path()
# ---------------------------------------------------------------------------

class TestResolveOutputPath:

    def test_explicit_parquet_out_returned_as_is(self, base_cfg):
        base_cfg.data_source = "parquet"
        base_cfg.parquet_out = "output/my_results.parquet"
        result = _resolve_output_path(base_cfg, "20250101T120000")
        assert result == "output/my_results.parquet"

    def test_auto_generated_path_contains_stem(self, base_cfg, tmp_path):
        base_cfg.data_source = "parquet"
        base_cfg.parquet_out = None
        base_cfg.parquet_in  = str(tmp_path / "emails_2024.parquet")
        ts     = "20250101T120000"
        result = _resolve_output_path(base_cfg, ts)
        assert "emails_2024" in result
        assert ts in result
        assert result.endswith(".parquet")

    def test_sql_mode_uses_table_name(self, base_cfg):
        base_cfg.data_source = "sql"
        base_cfg.sql_table   = "[dbo].[Original_Email]"
        base_cfg.parquet_out = None
        ts     = "20250101T120000"
        result = _resolve_output_path(base_cfg, ts)
        assert result.endswith(".parquet")
        assert ts in result

    def test_sql_mode_explicit_parquet_out(self, base_cfg):
        base_cfg.data_source = "sql"
        base_cfg.sql_table   = "[dbo].[Original_Email]"
        base_cfg.parquet_out = "output/sql_out.parquet"
        result = _resolve_output_path(base_cfg, "20250101T120000")
        assert result == "output/sql_out.parquet"


# ---------------------------------------------------------------------------
# _classify_row()
# ---------------------------------------------------------------------------

_DUMMY_PROMPT_INFO = {
    "system_prompt":     "sys",
    "user_prompt":       "usr",
    "prompt_tokens":     10,
    "completion_tokens": 5,
    "total_tokens":      15,
    "confidence":        "high",
    "all_intents":       ["Billing"],
}


class TestClassifyRowEmpty:
    """PATH A — empty subject and body should skip LLM."""

    def test_empty_both_returns_skipped(self, base_cfg):
        log = MagicMock()
        result = _classify_row(
            idx=0, guid="g1", subject="", body="",
            model=None, system_prompt="sys",
            valid_labels=["Billing"], cfg=base_cfg, log=log,
        )
        assert result["label"] == "Unclassified"
        assert result["processing_status"] == STATUS_SKIPPED_EMPTY
        assert result["llm_called"] is False

    def test_whitespace_only_is_not_skipped(self, base_cfg):
        """Whitespace-only subject/body is truthy — LLM is called."""
        log = MagicMock()
        with patch("classifier.pipeline.predict_intent") as mock_pi:
            mock_pi.return_value = ("Billing", "{}", STATUS_OK, _DUMMY_PROMPT_INFO)
            result = _classify_row(
                idx=0, guid="g1", subject="   ", body="   ",
                model=MagicMock(), system_prompt="sys",
                valid_labels=["Billing"], cfg=base_cfg, log=log,
            )
        assert result["llm_called"] is True


class TestClassifyRowNormal:
    """PATH B — normal inference."""

    def test_returns_label_from_predict_intent(self, base_cfg):
        log = MagicMock()
        with patch("classifier.pipeline.predict_intent") as mock_pi:
            mock_pi.return_value = ("Billing", "{}", STATUS_OK, _DUMMY_PROMPT_INFO)
            result = _classify_row(
                idx=0, guid="g1", subject="My bill", body="Overcharged",
                model=MagicMock(), system_prompt="sys",
                valid_labels=["Billing"], cfg=base_cfg, log=log,
            )
        assert result["label"] == "Billing"
        assert result["processing_status"] == STATUS_OK
        assert result["llm_called"] is True
        assert result["latency_ms"] >= 0

    def test_latency_is_non_negative(self, base_cfg):
        log = MagicMock()
        with patch("classifier.pipeline.predict_intent") as mock_pi:
            mock_pi.return_value = ("Outage", "{}", STATUS_OK, _DUMMY_PROMPT_INFO)
            result = _classify_row(
                idx=1, guid="g2", subject="No power", body="Blackout",
                model=MagicMock(), system_prompt="sys",
                valid_labels=["Outage"], cfg=base_cfg, log=log,
            )
        assert result["latency_ms"] >= 0


class TestClassifyRowException:
    """PATH C — exception during inference."""

    def test_exception_returns_error_status(self, base_cfg):
        log = MagicMock()
        with patch("classifier.pipeline.predict_intent", side_effect=RuntimeError("timeout")):
            result = _classify_row(
                idx=0, guid="g1", subject="My bill", body="Help",
                model=MagicMock(), system_prompt="sys",
                valid_labels=["Billing"], cfg=base_cfg, log=log,
            )
        assert result["label"] == "Unclassified"
        assert result["processing_status"] == STATUS_ERROR
        assert result["llm_called"] is True
        assert "timeout" in result["error_detail"]
