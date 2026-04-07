#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hydro One Email Intent Classifier — CLI entry point (v4.7.0)

All business logic lives in the `classifier/` package:
  classifier/config.py         — Config dataclass + constants
  classifier/utils.py          — PII redaction, hashing, file utils
  classifier/logger.py         — Structured/JSON logger
  classifier/kb.py             — Knowledgebase loader (Excel -> labels + terms)
  classifier/prompt_builder.py — PromptBudget, system prompt, text truncation
  classifier/model_loader.py   — GGUF model loader (llama.cpp best practices)
  classifier/inference.py      — LLM call, response parser, retry wrapper
  classifier/data_io.py        — Parquet + SQL Server I/O helpers
  classifier/telemetry.py      — Per-row inference records, manifest, integrity
  classifier/pipeline.py       — Full run() orchestrator

To run via YAML config use main.py instead.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

# llama_cpp must be imported before pandas on this machine (numpy DLL load order)
from classifier.config import DEFAULT_OUT_COL, Config, _PSUTIL_AVAILABLE
from classifier.pipeline import run
from classifier.utils import ensure_dir

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


# --------------------------------------------------
# CLI
# --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hydro One Email Intent Classifier v4.7.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group("Required")
    req.add_argument("--in",    dest="parquet_in", default="",
                     help="Input Parquet file (must contain GUID, Subject, Body). Not needed when data_source=sql.")
    req.add_argument("--model", dest="model_path", required=True,
                     help="Path to GGUF model file.")
    req.add_argument("--kb",    dest="kb_file",    required=True,
                     help="Path to Knowledgebase Excel file (sheet: Merged_Knowledgebase).")

    out = p.add_argument_group("Output")
    out.add_argument("--out",        dest="parquet_out", default=None,
                     help="Output Parquet path. Auto-named if omitted.")
    out.add_argument("--out-col",    dest="out_col",     default=DEFAULT_OUT_COL)
    out.add_argument("--record-dir", dest="record_dir",  default=None)
    out.add_argument("--no-records", dest="no_records",  action="store_true")

    llm = p.add_argument_group("LLM Settings")
    llm.add_argument("--n-ctx",          dest="n_ctx",          type=int,   default=8192)
    llm.add_argument("--n-gpu-layers",   dest="n_gpu_layers",   type=int,   default=0)
    llm.add_argument("--n-threads",      dest="n_threads",      type=int,
                     default=(_PSUTIL_AVAILABLE and psutil and psutil.cpu_count(logical=False) or os.cpu_count() or 4),
                     help="Physical CPU cores. Do not use logical/hyperthreaded count.")
    llm.add_argument("--n-batch",        dest="n_batch",        type=int,   default=512)
    llm.add_argument("--max-tokens",     dest="max_tokens",     type=int,   default=256)
    llm.add_argument("--temperature",    dest="temperature",    type=float, default=0.05)
    llm.add_argument("--top-p",          dest="top_p",          type=float, default=0.90,
                     help="Nucleus sampling. 0.90 recommended for classification.")
    llm.add_argument("--top-k",          dest="top_k",          type=int,   default=10,
                     help="Top-k sampling. 10 recommended for deterministic JSON output.")
    llm.add_argument("--repeat-penalty", dest="repeat_penalty", type=float, default=1.1)
    llm.add_argument("--max-keywords",   dest="max_keywords",   type=int,   default=5)

    gguf = p.add_argument_group("GGUF Best Practices")
    gguf.add_argument("--use-mlock",  dest="use_mlock",  action="store_true",
                      help="Pin model weights in RAM. Requires free RAM >= model size.")
    gguf.add_argument("--flash-attn", dest="flash_attn", action="store_true",
                      help="Enable flash attention (20-40%% faster on AVX2/AVX512 CPUs).")

    inf = p.add_argument_group("Inference Control")
    inf.add_argument("--timeout", dest="infer_timeout_sec", type=int, default=60)
    inf.add_argument("--retries", dest="infer_retries",     type=int, default=2)

    lg = p.add_argument_group("Logging")
    lg.add_argument("--log-file",    dest="log_file",    default=None)
    lg.add_argument("--log-level",   dest="log_level",   default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    lg.add_argument("--log-json",    dest="json_logs",   action="store_true")
    lg.add_argument("--no-redact",   dest="no_redact",   action="store_true")
    lg.add_argument("--log-prompts", dest="log_prompts", action="store_true")

    p.add_argument("--mode", dest="run_mode",
                   choices=["dev", "test", "prod"], default="prod")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    log_file = args.log_file
    if not log_file:
        _ref_path = args.parquet_out or args.parquet_in or "."
        logs_dir  = os.path.join(os.path.dirname(_ref_path) or "logs", "logs")
        ensure_dir(logs_dir)
        log_file  = os.path.join(
            logs_dir,
            f"email_intent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

    run(Config(
        model_path        = args.model_path,
        kb_file           = args.kb_file,
        parquet_in        = args.parquet_in,
        parquet_out       = args.parquet_out,
        out_col           = args.out_col,
        record_dir        = args.record_dir,
        log_file          = log_file,
        n_ctx             = args.n_ctx,
        n_gpu_layers      = args.n_gpu_layers,
        n_threads         = args.n_threads,
        n_batch           = args.n_batch,
        max_tokens        = args.max_tokens,
        temperature       = args.temperature,
        top_p             = args.top_p,
        top_k             = args.top_k,
        repeat_penalty    = args.repeat_penalty,
        max_keywords      = args.max_keywords,
        use_mlock         = args.use_mlock,
        flash_attn        = args.flash_attn,
        infer_timeout_sec = args.infer_timeout_sec,
        infer_retries     = args.infer_retries,
        run_mode          = args.run_mode,
        log_level         = args.log_level,
        json_logs         = args.json_logs,
        redact_logs       = not args.no_redact,
        no_records        = args.no_records,
        log_prompts       = args.log_prompts,
    ))


if __name__ == "__main__":
    main()