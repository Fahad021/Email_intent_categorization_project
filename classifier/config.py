from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# --------------------------------------------------
# VERSION
# --------------------------------------------------

APP_VERSION     = "4.7.0"
DEFAULT_OUT_COL = "Predicted_Reduced_Category"

# Prompt budget constants
SYSTEM_PROMPT_MAX_RATIO = 0.40
OUTPUT_RESERVE_TOKENS   = 256
OVERHEAD_TOKENS         = 60
CHARS_PER_TOKEN         = 3.5

# Processing status values — full audit trail per row
STATUS_OK               = "ok"               # LLM called, valid category returned
STATUS_CASE_CORRECTED   = "case_corrected"   # LLM called, category matched after normalisation
STATUS_NO_JSON          = "no_json"          # LLM called, response had no JSON
STATUS_BAD_JSON         = "bad_json"         # LLM called, JSON malformed
STATUS_EMPTY_FIELDS     = "empty_fields"     # LLM called, JSON had no category fields
STATUS_INVALID_CATEGORY = "invalid_category" # LLM called, category not in allowed list
STATUS_SKIPPED_EMPTY    = "skipped_empty"    # NOT called — subject+body both empty
STATUS_ERROR            = "error"            # NOT called — exception during inference
STATUS_UNKNOWN          = "unknown"          # fallback


# --------------------------------------------------
# CONFIG DATACLASS
# --------------------------------------------------

@dataclass
class Config:
    # Required paths
    model_path: str
    kb_file:    str

    # Prompt file (leave blank to use built-in hardcoded prompt)
    prompt_file: str = ""

    # Data source: "parquet" | "sql"
    data_source: str = "parquet"

    # Parquet I/O (used when data_source = "parquet")
    parquet_in:  str           = ""
    parquet_out: Optional[str] = None
    out_col:     str           = DEFAULT_OUT_COL

    # SQL Server (used when data_source = "sql")
    sql_server:   str  = ""
    sql_database: str  = "EMAIL"
    sql_table:    str  = "[dbo].[Original_Email]"
    sql_trusted:  bool = True
    sql_batch:    int  = 100
    sql_dry_run:  bool = False

    # Optional paths
    record_dir: Optional[str] = None
    log_file:   Optional[str] = None

    # LLM behaviour
    n_ctx:          int   = 8192
    n_gpu_layers:   int   = 0
    n_threads:      int   = field(default_factory=lambda: (
        psutil.cpu_count(logical=False) if _PSUTIL_AVAILABLE else (os.cpu_count() or 4)
    ))
    n_batch:        int   = 512
    max_tokens:     int   = 256
    temperature:    float = 0.05
    top_p:          float = 0.90
    top_k:          int   = 10
    repeat_penalty: float = 1.1
    max_keywords:   int   = 5

    # GGUF best practices
    use_mlock:   bool = False
    flash_attn:  bool = False

    # Inference control
    infer_timeout_sec: int = 60
    infer_retries:     int = 2

    # Run behaviour
    run_mode:    str  = "prod"
    run_id:      str  = field(default_factory=lambda: uuid.uuid4().hex[:8])
    log_level:   str  = "INFO"
    json_logs:   bool = False
    redact_logs: bool = True
    no_records:  bool = False
    log_prompts: bool = False

    def __post_init__(self):
        self.run_mode    = self.run_mode.lower()
        self.log_level   = self.log_level.upper()
        self.data_source = self.data_source.lower()
        if self.run_mode not in ("dev", "test", "prod"):
            raise ValueError(f"run_mode must be dev/test/prod, got: {self.run_mode}")
        if self.data_source not in ("parquet", "sql"):
            raise ValueError(f"data_source must be 'parquet' or 'sql', got: {self.data_source}")
        if self.data_source == "parquet" and not self.parquet_in:
            raise ValueError("parquet_in is required when data_source = 'parquet'")
        if self.data_source == "sql" and not self.sql_server:
            raise ValueError("sql_server is required when data_source = 'sql'")
