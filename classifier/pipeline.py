from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .config import Config, STATUS_ERROR, STATUS_SKIPPED_EMPTY
from .data_io import read_from_sql, read_parquet, write_parquet, write_to_sql
from .inference import log_prompt, predict_intent
from .kb import load_reduced_kb
from .logger import build_logger
from .model_loader import load_model
from .prompt_builder import PromptBudget, build_system_prompt, prepare_subject_body
from .telemetry import build_integrity_report, write_inference_record, write_manifest, write_prompt_record
from .utils import ensure_dir, redact


# --------------------------------------------------
# OUTPUT PATH RESOLVER
# --------------------------------------------------

def _resolve_output_path(cfg: Config, ts: str) -> str:
    """Determine the output parquet path from config and a timestamp string."""
    if cfg.data_source == "sql":
        return cfg.parquet_out or os.path.join(
            "output", f"sql_{cfg.sql_table}_{ts}.parquet"
        )
    if cfg.parquet_out:
        return cfg.parquet_out
    stem = os.path.splitext(os.path.basename(cfg.parquet_in))[0]
    return os.path.join(
        os.path.dirname(cfg.parquet_in),
        f"{stem}_with_llm_predictions_{ts}.parquet",
    )


# --------------------------------------------------
# ROW CLASSIFIER
# --------------------------------------------------

def _classify_row(
    idx:           int,
    guid:          str,
    subject:       str,
    body:          str,
    model,
    system_prompt: str,
    valid_labels:  List[str],
    cfg:           Config,
    log:           logging.Logger,
) -> Dict:
    """
    Classify a single email row and return a result dict.

    Three paths:
      PATH A - empty input: skip LLM, status = skipped_empty
      PATH B - normal:      call LLM, parse response
      PATH C - exception:   status = error, llm_called = True

    Keys in the returned dict:
      label, raw, parse_stat, processing_status,
      llm_called, error_detail, prompt_info, latency_ms
    """
    t0 = time.time()
    _empty_prompt: Dict = {
        "system_prompt":     system_prompt,
        "user_prompt":       "",
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "total_tokens":      0,
        "confidence":        "low",
        "all_intents":       [],
    }

    # PATH A: empty input — skip LLM
    if not subject and not body:
        return {
            "label":             "Unclassified",
            "raw":               "",
            "parse_stat":        STATUS_SKIPPED_EMPTY,
            "processing_status": STATUS_SKIPPED_EMPTY,
            "llm_called":        False,
            "error_detail":      "",
            "prompt_info":       _empty_prompt,
            "latency_ms":        int((time.time() - t0) * 1000),
        }

    # PATH B: normal — call LLM
    try:
        label, raw, parse_stat, prompt_info = predict_intent(
            model=model, system_prompt=system_prompt,
            subject=subject, body=body,
            valid_labels=valid_labels, cfg=cfg, log=log,
        )
        return {
            "label":             label,
            "raw":               raw,
            "parse_stat":        parse_stat,
            "processing_status": parse_stat,
            "llm_called":        True,
            "error_detail":      "",
            "prompt_info":       prompt_info,
            "latency_ms":        int((time.time() - t0) * 1000),
        }

    # PATH C: exception during inference
    except Exception as e:
        error_detail = str(e)
        log.error("inference_error", extra={"guid": guid, "row_index": idx, "error": error_detail})
        return {
            "label":             "Unclassified",
            "raw":               "",
            "parse_stat":        STATUS_ERROR,
            "processing_status": STATUS_ERROR,
            "llm_called":        True,
            "error_detail":      error_detail,
            "prompt_info":       _empty_prompt,
            "latency_ms":        int((time.time() - t0) * 1000),
        }


# --------------------------------------------------
# TELEMETRY RECORD WRITER
# --------------------------------------------------

def _write_row_records(
    result:       Dict,
    idx:          int,
    guid:         str,
    subject:      str,
    body:         str,
    records_base: str,
    prompts_dir:  str,
    file_in:      str,
    budget:       PromptBudget,
    cfg:          Config,
    log:          logging.Logger,
) -> None:
    """Write per-row inference and prompt telemetry records. Silently logs failures."""
    prompt_info = result["prompt_info"]
    try:
        write_inference_record(
            out_dir=records_base,
            guid=guid,
            row_index=idx,
            subject=subject,
            body=body,
            user_prompt=prompt_info["user_prompt"],
            system_prompt=prompt_info["system_prompt"],
            raw_response=result["raw"],
            label=result["label"],
            parse_status=result["parse_stat"],
            processing_status=result["processing_status"],
            llm_called=result["llm_called"],
            latency_ms=result["latency_ms"],
            parquet_in=file_in,
            prompt_tokens=prompt_info["prompt_tokens"],
            completion_tokens=prompt_info["completion_tokens"],
            total_tokens=prompt_info["total_tokens"],
            confidence=prompt_info.get("confidence", "unknown"),
            all_intents=prompt_info.get("all_intents", []),
            body_budget=budget.body_char_limit,
            cfg=cfg,
        )
        if result["llm_called"]:
            write_prompt_record(
                out_dir=prompts_dir,
                row_index=idx,
                guid=guid,
                system_prompt=prompt_info["system_prompt"],
                user_prompt=prompt_info["user_prompt"],
                prompt_tokens=prompt_info["prompt_tokens"],
                completion_tokens=prompt_info["completion_tokens"],
                total_tokens=prompt_info["total_tokens"],
                cfg=cfg,
            )
    except Exception as e:
        log.warning("record_write_failed", extra={"guid": guid, "row_index": idx, "msg": str(e)})


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run(cfg: Config) -> None:
    """
    Full classification pipeline:
      1. Build logger
      2. Load KB + build system prompt
      3. Load model
      4. Compute prompt budget
      5. Read input (parquet or SQL)
      6. Inference loop (every row explicitly tracked)
      7. Data integrity check
      8. Write output parquet (with audit columns)
      9. Write manifest
    """
    log = build_logger(cfg)

    # Step 2 — KB + prompt
    valid_labels, category_terms = load_reduced_kb(cfg.kb_file, log)
    system_prompt = build_system_prompt(
        valid_labels, category_terms, cfg.max_keywords, cfg.prompt_file
    )
    log.info("system_prompt_built", extra={
        "prompt_chars": len(system_prompt),
        "prompt_file":  cfg.prompt_file or "<built-in>",
    })
    if cfg.log_prompts or cfg.run_mode in ("dev", "test"):
        log_prompt("system", system_prompt, cfg, log)

    # Step 3 — Model
    model = load_model(cfg, log)

    # Step 4 — Prompt budget
    budget = PromptBudget(model, system_prompt, cfg, log)
    log.info("body_budget_set", extra={
        "body_char_limit": budget.body_char_limit,
        "system_tokens":   budget.system_tokens,
    })

    # Step 5 — Input
    if cfg.data_source == "sql":
        df       = read_from_sql(cfg, log)
        _file_in = f"sql://{cfg.sql_server}/{cfg.sql_database}/{cfg.sql_table}"
    else:
        df       = read_parquet(cfg.parquet_in)
        _file_in = cfg.parquet_in

    missing = [c for c in ("GUID", "Subject", "Body") if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    rows_in = len(df)
    log.info("input_loaded", extra={"file_in": _file_in, "rows_in": rows_in})

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_out = _resolve_output_path(cfg, ts)

    records_base = cfg.record_dir or os.path.join(os.path.dirname(parquet_out), "inference_records")
    prompts_dir  = os.path.join(records_base, "prompts")
    if not cfg.no_records:
        ensure_dir(records_base)
        ensure_dir(prompts_dir)

    # Step 6 — Inference loop
    predictions:         List[str]  = []
    parse_statuses:      List[str]  = []
    processing_statuses: List[str]  = []
    llm_called_flags:    List[bool] = []
    confidences:         List[str]  = []
    all_intents_col:     List[str]  = []
    error_details:       List[str]  = []

    t_run_start = time.time()

    for idx, row in tqdm(df.iterrows(), total=rows_in, desc="Classifying", leave=True):
        guid    = str(row.get("GUID",    ""))
        subject = str(row.get("Subject") or "")
        body    = str(row.get("Body")    or "")

        subject, body = prepare_subject_body(subject, body, budget)
        result        = _classify_row(idx, guid, subject, body, model, system_prompt, valid_labels, cfg, log)
        prompt_info   = result["prompt_info"]

        predictions.append(result["label"])
        parse_statuses.append(result["parse_stat"])
        processing_statuses.append(result["processing_status"])
        llm_called_flags.append(result["llm_called"])
        confidences.append(prompt_info.get("confidence", "unknown"))
        all_intents_col.append(json.dumps(prompt_info.get("all_intents", [])))
        error_details.append(result["error_detail"])

        log.info(
            "inference_done",
            extra={
                "guid":              guid,
                "row_index":         idx,
                "latency_ms":        result["latency_ms"],
                "intent_code":       result["label"],
                "intent":            result["label"],
                "subject_len":       len(subject),
                "body_len":          len(body),
                "body_budget":       budget.body_char_limit,
                "response_raw":      redact(result["raw"], enabled=cfg.redact_logs, max_chars=500),
                "parse_status":      result["parse_stat"],
                "processing_status": result["processing_status"],
                "llm_called":        result["llm_called"],
                "confidence":        prompt_info.get("confidence", "unknown"),
                "file_in":           _file_in,
                "model":             os.path.basename(cfg.model_path),
                "model_ctx":         cfg.n_ctx,
                "max_tokens":        cfg.max_tokens,
                "temperature":       cfg.temperature,
                "n_threads":         cfg.n_threads,
                "n_batch":           cfg.n_batch,
                "prompt_tokens":     prompt_info["prompt_tokens"],
                "completion_tokens": prompt_info["completion_tokens"],
                "total_tokens":      prompt_info["total_tokens"],
            },
        )

        if not cfg.no_records:
            _write_row_records(result, idx, guid, subject, body, records_base, prompts_dir, _file_in, budget, cfg, log)

    # Step 7 — Integrity check
    integrity_report = build_integrity_report(
        df_in=df,
        predictions=predictions,
        llm_called_flags=llm_called_flags,
        processing_statuses=processing_statuses,
        parquet_out=parquet_out,
        log=log,
    )

    # Step 8 — Write output
    df[cfg.out_col]        = predictions
    df["Parse_Status"]     = parse_statuses
    df["Processing_Status"] = processing_statuses
    df["LLM_Called"]       = llm_called_flags
    df["Confidence"]       = confidences
    df["All_Intents"]      = all_intents_col
    df["Error_Detail"]     = error_details

    write_parquet(df, parquet_out)
    if cfg.data_source == "sql":
        write_to_sql(df, cfg, log)

    log.info("output_written", extra={"file_out": parquet_out, "rows_total": len(df)})

    # Step 9 — Manifest
    write_manifest(
        parquet_in=_file_in,
        parquet_out=parquet_out,
        log_file=cfg.log_file,
        predictions=predictions,
        integrity_report=integrity_report,
        cfg=cfg,
        log=log,
    )

    log.info(
        "run_complete",
        extra={
            "rows_total":       len(df),
            "duration_s":       round(time.time() - t_run_start, 3),
            "file_in":          _file_in,
            "file_out":         parquet_out,
            "integrity_passed": integrity_report["passed"],
        },
    )

    if not integrity_report["passed"]:
        raise RuntimeError(
            f"Data integrity check FAILED. See manifest at {parquet_out}.manifest.json"
        )
