# tests/conftest.py
"""
Shared pytest fixtures used across unit and integration tests.
"""
from __future__ import annotations

import os
import sys
from typing import List
from unittest.mock import MagicMock

# IMPORTANT: llama_cpp must be imported before pandas/numpy to avoid a DLL
# conflict on Windows.  This mirrors the constraint in claude.py / pipeline.py.
from llama_cpp import Llama  # noqa: F401  (import for side-effect only)

import pandas as pd
import pytest

# Ensure project root is importable regardless of working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from classifier.config import Config


# ---------------------------------------------------------------------------
# Minimal valid Config (parquet mode, no real files needed for unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_cfg(tmp_path) -> Config:
    """A fully valid Config that won't trigger __post_init__ errors."""
    parquet_in = str(tmp_path / "emails.parquet")
    # write a placeholder so path validation (if any) succeeds
    pd.DataFrame({"GUID": [], "Subject": [], "Body": []}).to_parquet(parquet_in, engine="pyarrow")
    return Config(
        model_path  = str(tmp_path / "model.gguf"),
        kb_file     = str(tmp_path / "kb.xlsx"),
        parquet_in  = parquet_in,
        data_source = "parquet",
        run_mode    = "test",
        no_records  = True,
    )


# ---------------------------------------------------------------------------
# Mock Llama model
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llama():
    """
    A MagicMock that mimics llama_cpp.Llama well enough for unit tests.
    - tokenize()  → list of ints (length ≈ chars / 3.5)
    - __call__()  → response dict with 'choices'
    """
    m = MagicMock()
    m.tokenize.side_effect = lambda b: list(range(max(1, len(b) // 4)))
    m.return_value = {
        "choices": [{"text": '{"intent_category": "Billing", "confidence": "high", "all_intents": ["Billing"]}'}],
        "usage":   {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    return m


# ---------------------------------------------------------------------------
# Sample labels and category_terms
# ---------------------------------------------------------------------------

SAMPLE_LABELS: List[str] = ["Billing", "Outage", "New Connection", "General Inquiry"]

@pytest.fixture()
def sample_labels() -> List[str]:
    return list(SAMPLE_LABELS)


# ---------------------------------------------------------------------------
# Sample DataFrame of emails
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_email_df() -> pd.DataFrame:
    return pd.DataFrame({
        "GUID":    ["guid-001", "guid-002", "guid-003", "guid-004"],
        "Subject": ["My bill is wrong",  "Power is out", "",          "Hello"],
        "Body":    ["I was overcharged", "No power",    "",          ""],
    })
