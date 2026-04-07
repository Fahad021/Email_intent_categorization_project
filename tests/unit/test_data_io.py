# tests/unit/test_data_io.py
"""
Unit tests for classifier.data_io

Parquet write/read round-trip using pyarrow only — no SQL Server needed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from classifier.data_io import read_parquet, write_parquet


SAMPLE_DF = pd.DataFrame({
    "GUID":    ["g1", "g2", "g3"],
    "Subject": ["Bill query", "Outage report", ""],
    "Body":    ["I was overcharged", "No power since 8am", ""],
    "Label":   ["Billing", "Outage", "Unclassified"],
})


class TestParquetRoundTrip:

    def test_write_then_read_preserves_shape(self, tmp_path):
        path = str(tmp_path / "test.parquet")
        write_parquet(SAMPLE_DF, path)
        df_back = read_parquet(path)
        assert df_back.shape == SAMPLE_DF.shape

    def test_write_then_read_preserves_values(self, tmp_path):
        path = str(tmp_path / "test.parquet")
        write_parquet(SAMPLE_DF, path)
        df_back = read_parquet(path)
        assert list(df_back["GUID"]) == list(SAMPLE_DF["GUID"])
        assert list(df_back["Label"]) == list(SAMPLE_DF["Label"])

    def test_write_then_read_preserves_columns(self, tmp_path):
        path = str(tmp_path / "test.parquet")
        write_parquet(SAMPLE_DF, path)
        df_back = read_parquet(path)
        assert set(df_back.columns) == set(SAMPLE_DF.columns)

    def test_empty_dataframe_round_trip(self, tmp_path):
        path = str(tmp_path / "empty.parquet")
        empty = pd.DataFrame({"GUID": [], "Subject": [], "Body": []})
        write_parquet(empty, path)
        df_back = read_parquet(path)
        assert len(df_back) == 0
        assert list(df_back.columns) == ["GUID", "Subject", "Body"]

    def test_read_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            read_parquet(str(tmp_path / "missing.parquet"))
