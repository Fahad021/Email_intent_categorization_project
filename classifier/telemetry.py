from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .config import APP_VERSION, Config, STATUS_BAD_JSON, STATUS_CASE_CORRECTED, STATUS_EMPTY_FIELDS, STATUS_ERROR, STATUS_INVALID_CATEGORY, STATUS_NO_JSON, STATUS_OK, STATUS_SKIPPED_EMPTY
from .utils import ensure_dir, preview, redact, sha256_str


def write_inference_record(
    out_dir:            str,
    guid:               str,
    row_index:          int,
    subject:            str,
    body:               str,
    user_prompt:        str,
    system_prompt:      str,
    raw_response:       str,
    label:              str,
    parse_status:       str,
    processing_status:  str,
    llm_called:         bool,
    latency_ms:         int,
    parquet_in:         str,
    prompt_tokens:      int,
    completion_tokens:  int,
    total_tokens:       int,
    confidence:         str,
    all_intents:        List[str],
    body_budget:        int,
    cfg:                Config,
) -> None:
    ensure_dir(out_dir)
    ts        = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    safe_guid = (guid or "no_guid").replace("/", "_").strip()[:64]
    path      = os.path.join(out_dir, f"{row_index:06d}__{safe_guid}__{ts}.json")

    record: Dict = {
        "ts": ts, "run_id": cfg.run_id, "mode": cfg.run_mode,
        "app_version": APP_VERSION, "row_index": row_index, "guid": guid,
        "input": {
            "subject_len":    len(subject or ""),
            "body_len":       len(body    or ""),
            "body_budget":    body_budget,
            "subject_sha256": sha256_str(subject or ""),
            "body_sha256":    sha256_str(body    or ""),
        },
        "prompts": {
            "system_id":     sha256_str(system_prompt or ""),
            "user_id":       sha256_str(user_prompt   or ""),
            "prompt_tokens": prompt_tokens,
        },
        "response": {
            "text_sha256":       sha256_str(raw_response or ""),
            "chars":             len(raw_response or ""),
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
        },
        "prediction": {
            "intent_code":       label,
            "intent":            label,
            "parse_status":      parse_status,
            "processing_status": processing_status,
            "llm_called":        llm_called,
            "confidence":        confidence,
            "all_intents":       all_intents,
        },
        "timing": {"latency_ms": latency_ms},
        "model": {
            "file":           os.path.basename(cfg.model_path),
            "ctx":            cfg.n_ctx,
            "n_threads":      cfg.n_threads,
            "n_batch":        cfg.n_batch,
            "n_gpu_layers":   cfg.n_gpu_layers,
            "max_tokens":     cfg.max_tokens,
            "temperature":    cfg.temperature,
            "top_p":          cfg.top_p,
            "top_k":          cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "flash_attn":     cfg.flash_attn,
            "use_mlock":      cfg.use_mlock,
        },
        "source": {
            "parquet_in":  parquet_in,
            "app_version": APP_VERSION,
            "python":      sys.version.split()[0],
            "platform":    platform.platform(),
        },
    }

    if cfg.run_mode in ("dev", "test"):
        record["prompts"]["system_preview"] = redact(preview(system_prompt, 600), cfg.redact_logs)
        record["prompts"]["user_preview"]   = redact(preview(user_prompt,   600), cfg.redact_logs)
        record["response"]["preview"]       = redact(preview(raw_response,  600), cfg.redact_logs)

    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def write_prompt_record(
    out_dir:           str,
    row_index:         int,
    guid:              str,
    system_prompt:     str,
    user_prompt:       str,
    prompt_tokens:     int,
    completion_tokens: int,
    total_tokens:      int,
    cfg:               Config,
) -> None:
    ensure_dir(out_dir)
    ts        = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    safe_guid = (guid or "no_guid").replace("/", "_").strip()[:64]
    path      = os.path.join(out_dir, f"prompt_{row_index:06d}__{safe_guid}__{ts}.json")

    with io.open(path, "w", encoding="utf-8") as f:
        json.dump({
            "ts": ts, "run_id": cfg.run_id,
            "row_index": row_index, "guid": guid,
            "prompt_tokens":        prompt_tokens,
            "completion_tokens":    completion_tokens,
            "total_tokens":         total_tokens,
            "system_prompt":        system_prompt,
            "system_prompt_sha256": sha256_str(system_prompt or ""),
            "user_prompt":          user_prompt,
            "user_prompt_sha256":   sha256_str(user_prompt   or ""),
        }, f, ensure_ascii=False, indent=2)


def write_manifest(
    parquet_in:       str,
    parquet_out:      str,
    log_file:         Optional[str],
    predictions:      List[str],
    integrity_report: Dict,
    cfg:              Config,
    log:              logging.Logger,
) -> None:
    dist     = Counter(predictions)
    manifest = {
        "app_version": APP_VERSION,
        "timestamp":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S%fZ"),
        "run_id":      cfg.run_id,
        "mode":        cfg.run_mode,
        "python":      sys.version.split()[0],
        "platform":    platform.platform(),
        "in_file":     parquet_in,
        "out_file":    parquet_out,
        "log_file":    log_file,
        "rows":               len(predictions),
        "unclassified":       dist.get("Unclassified", 0),
        "distinct_intents":   len(dist),
        "intent_distribution": dist.most_common(),
        "integrity":   integrity_report,
        "model": {
            "file":           cfg.model_path,
            "ctx":            cfg.n_ctx,
            "n_threads":      cfg.n_threads,
            "n_batch":        cfg.n_batch,
            "n_gpu_layers":   cfg.n_gpu_layers,
            "max_tokens":     cfg.max_tokens,
            "temperature":    cfg.temperature,
            "top_p":          cfg.top_p,
            "top_k":          cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "flash_attn":     cfg.flash_attn,
            "use_mlock":      cfg.use_mlock,
        },
    }

    try:
        h = hashlib.sha256()
        with open(cfg.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        manifest["model"]["sha256"] = h.hexdigest()
    except Exception as e:
        manifest["model"]["sha256_error"] = str(e)

    manifest_path = parquet_out + ".manifest.json"
    with io.open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("manifest_written", extra={"file_out": manifest_path})


def build_integrity_report(
    df_in:               pd.DataFrame,
    predictions:         List[str],
    llm_called_flags:    List[bool],
    processing_statuses: List[str],
    parquet_out:         str,
    log:                 logging.Logger,
) -> Dict:
    """
    Reconcile every input row against every output row.
    Logs PASS or FAIL with a full breakdown.
    Returns the report dict (also embedded in the manifest).
    """
    rows_in          = len(df_in)
    rows_out         = len(predictions)
    rows_llm_called  = sum(llm_called_flags)
    rows_skipped     = sum(1 for s in processing_statuses if s == STATUS_SKIPPED_EMPTY)
    rows_errored     = sum(1 for s in processing_statuses if s == STATUS_ERROR)
    rows_ok          = sum(1 for s in processing_statuses if s in (STATUS_OK, STATUS_CASE_CORRECTED))
    rows_parse_issue = sum(
        1 for s in processing_statuses
        if s in (STATUS_NO_JSON, STATUS_BAD_JSON, STATUS_EMPTY_FIELDS, STATUS_INVALID_CATEGORY)
    )

    row_count_match    = rows_in == rows_out
    expected_llm_calls = rows_in - rows_skipped
    llm_call_match     = rows_llm_called == expected_llm_calls
    passed             = row_count_match and llm_call_match

    report = {
        "passed":              passed,
        "rows_in":             rows_in,
        "rows_out":            rows_out,
        "row_count_match":     row_count_match,
        "rows_llm_called":     rows_llm_called,
        "expected_llm_calls":  expected_llm_calls,
        "llm_call_match":      llm_call_match,
        "rows_ok":             rows_ok,
        "rows_case_corrected": sum(1 for s in processing_statuses if s == STATUS_CASE_CORRECTED),
        "rows_skipped_empty":  rows_skipped,
        "rows_errored":        rows_errored,
        "rows_parse_issues":   rows_parse_issue,
        "out_file":            parquet_out,
    }

    if passed:
        log.info(
            "integrity_check_passed",
            extra={
                "rows_in":         rows_in,
                "rows_out":        rows_out,
                "rows_llm_called": rows_llm_called,
                "rows_skipped":    rows_skipped,
                "rows_errored":    rows_errored,
                "rows_matched":    rows_ok,
            },
        )
    else:
        issues = []
        if not row_count_match:
            issues.append(f"ROW COUNT MISMATCH: in={rows_in} out={rows_out}")
        if not llm_call_match:
            issues.append(f"LLM CALL MISMATCH: expected={expected_llm_calls} actual={rows_llm_called}")
        log.error(
            "integrity_check_FAILED",
            extra={
                "rows_in":            rows_in,
                "rows_out":           rows_out,
                "rows_llm_called":    rows_llm_called,
                "expected_llm_calls": expected_llm_calls,
                "rows_skipped":       rows_skipped,
                "rows_errored":       rows_errored,
                "issues":             " | ".join(issues),
            },
        )

    return report
