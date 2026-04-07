from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import APP_VERSION, Config


def build_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("hydro_classifier")
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(cfg.log_level)
    run_id   = cfg.run_id
    run_mode = cfg.run_mode

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts":          self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level":       record.levelname,
                "logger":      record.name,
                "msg":         record.getMessage(),
                "run_id":      getattr(record, "run_id",      run_id),
                "mode":        getattr(record, "mode",        run_mode),
                "app_version": APP_VERSION,
            }
            for k in [
                "row_index", "latency_ms",
                "intent_code", "intent",
                "subject_len", "body_len",
                "prompt_chars", "prompt_id",
                "response_chars", "response_raw",
                "parse_status", "processing_status", "llm_called",
                "file_in", "file_out",
                "model", "model_ctx",
                "max_tokens", "temperature",
                "n_threads", "n_batch", "n_gpu_layers",
                "prompt_tokens", "completion_tokens", "total_tokens",
                "confidence", "body_budget",
                "rows_in", "rows_out", "rows_llm_called",
                "rows_skipped", "rows_errored", "rows_matched",
            ]:
                if hasattr(record, k):
                    payload[k] = getattr(record, k)
            return json.dumps(payload, ensure_ascii=False)

    class ContextFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "run_id"):      record.run_id      = run_id
            if not hasattr(record, "mode"):        record.mode        = run_mode
            if not hasattr(record, "app_version"): record.app_version = APP_VERSION
            return True

    fmt = (
        JsonFormatter()
        if cfg.json_logs
        else logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.addFilter(ContextFilter())
    logger.addHandler(ch)

    if cfg.log_file:
        fh = RotatingFileHandler(
            cfg.log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        fh.addFilter(ContextFilter())
        logger.addHandler(fh)

    logger._configured = True
    return logger
