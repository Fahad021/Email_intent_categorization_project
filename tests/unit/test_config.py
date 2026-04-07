# tests/unit/test_config.py
"""
Unit tests for classifier.config.Config

Tests cover:
- Valid construction (defaults + overrides)
- __post_init__ normalisation (lowercasing run_mode, data_source, log_level)
- Validation errors (bad run_mode, bad data_source, missing parquet_in, missing sql_server)
"""
from __future__ import annotations

import pytest

from classifier.config import Config, DEFAULT_OUT_COL


def _parquet_cfg(**overrides) -> Config:
    defaults = dict(
        model_path  = "models/model.gguf",
        kb_file     = "data/kb.xlsx",
        parquet_in  = "data/emails.parquet",
        data_source = "parquet",
        run_mode    = "test",
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_minimal_parquet_mode(self):
        cfg = _parquet_cfg()
        assert cfg.model_path == "models/model.gguf"
        assert cfg.data_source == "parquet"

    def test_defaults_applied(self):
        cfg = _parquet_cfg()
        assert cfg.out_col == DEFAULT_OUT_COL
        assert cfg.n_ctx == 8192
        assert cfg.temperature == pytest.approx(0.05)
        assert cfg.redact_logs is True

    def test_run_id_generated(self):
        cfg = _parquet_cfg()
        assert isinstance(cfg.run_id, str) and len(cfg.run_id) == 8

    def test_run_id_unique_per_instance(self):
        a = _parquet_cfg()
        b = _parquet_cfg()
        assert a.run_id != b.run_id


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:

    def test_run_mode_lowercased(self):
        cfg = _parquet_cfg(run_mode="TEST")
        assert cfg.run_mode == "test"

    def test_log_level_uppercased(self):
        cfg = _parquet_cfg(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_data_source_lowercased(self):
        cfg = _parquet_cfg(data_source="PARQUET")
        assert cfg.data_source == "parquet"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:

    def test_invalid_run_mode_raises(self):
        with pytest.raises(ValueError, match="run_mode"):
            _parquet_cfg(run_mode="staging")

    def test_invalid_data_source_raises(self):
        with pytest.raises(ValueError, match="data_source"):
            Config(
                model_path  = "m.gguf",
                kb_file     = "kb.xlsx",
                data_source = "csv",
                parquet_in  = "x.parquet",
            )

    def test_parquet_mode_requires_parquet_in(self):
        with pytest.raises(ValueError, match="parquet_in"):
            Config(
                model_path  = "m.gguf",
                kb_file     = "kb.xlsx",
                data_source = "parquet",
                parquet_in  = "",
            )

    def test_sql_mode_requires_sql_server(self):
        with pytest.raises(ValueError, match="sql_server"):
            Config(
                model_path  = "m.gguf",
                kb_file     = "kb.xlsx",
                data_source = "sql",
                sql_server  = "",
            )
