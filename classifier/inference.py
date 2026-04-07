from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Optional, Tuple

from llama_cpp import Llama

from .config import (
    Config,
    OUTPUT_RESERVE_TOKENS,
    STATUS_BAD_JSON,
    STATUS_CASE_CORRECTED,
    STATUS_EMPTY_FIELDS,
    STATUS_INVALID_CATEGORY,
    STATUS_NO_JSON,
    STATUS_OK,
)
from .utils import preview, redact, sha256_str


# --------------------------------------------------
# PROMPT / RESPONSE TELEMETRY
# --------------------------------------------------

def log_prompt(kind: str, text: str, cfg: Config, log: logging.Logger) -> None:
    extra: Dict = {
        "prompt_kind":  kind,
        "prompt_id":    sha256_str(text),
        "prompt_chars": len(text),
    }
    if cfg.run_mode in ("dev", "test"):
        extra["prompt_preview"] = redact(preview(text, 400), enabled=cfg.redact_logs)
    log.debug("prompt_built", extra=extra)


def log_response(text: str, cfg: Config, log: logging.Logger) -> None:
    extra: Dict = {
        "response_id":    sha256_str(text or ""),
        "response_chars": len(text or ""),
    }
    if cfg.run_mode in ("dev", "test"):
        extra["response_preview"] = redact(preview(text, 400), enabled=cfg.redact_logs)
    log.debug("model_response", extra=extra)


# --------------------------------------------------
# RETRY WRAPPER
# --------------------------------------------------

def call_with_retries(fn, timeout_s: int, retries: int):
    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(fn).result(timeout=timeout_s)
        except (TimeoutError, Exception) as e:
            last_exc = e
            if attempt <= retries:
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
    raise last_exc


# --------------------------------------------------
# RESPONSE PARSER
# --------------------------------------------------

def parse_llm_output(
    raw:          str,
    valid_labels: List[str],
) -> Tuple[str, str, str, str, List[str]]:
    """
    Parse LLM output JSON into (label, raw, parse_status, confidence, all_intents).
    parse_status values map directly to STATUS_* constants.
    """
    raw = (raw or "").strip()

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    cleaned = re.sub(r"```", "", cleaned).strip()

    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        try:
            obj = json.loads(cleaned)
        except Exception:
            return "Unclassified", raw, STATUS_NO_JSON, "unknown", []
    else:
        raw_json = re.sub(r",\s*(\})", r"\1", match.group())
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            return "Unclassified", raw, STATUS_BAD_JSON, "unknown", []

    code  = str(obj.get("intent_code")     or obj.get("intent_category") or "").strip()
    label = str(obj.get("intent_category") or obj.get("intent_code")     or "").strip()
    confidence      = str(obj.get("confidence") or "unknown").strip().lower()
    all_intents_raw = obj.get("all_intents", [])

    if confidence not in ("high", "medium", "low"):
        confidence = "unknown"

    all_intents = (
        [str(i).strip() for i in all_intents_raw if i]
        if isinstance(all_intents_raw, list)
        else ([str(all_intents_raw).strip()] if all_intents_raw else [])
    )

    if not code and not label:
        return "Unclassified", raw, STATUS_EMPTY_FIELDS, confidence, all_intents

    valid_lower = {v.lower(): v for v in valid_labels}

    def resolve(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        if candidate in valid_labels:
            return candidate
        if candidate.lower() in valid_lower:
            return valid_lower[candidate.lower()]
        stripped = re.sub(r"[^\w\s&]", "", candidate).strip()
        return valid_lower.get(stripped.lower())

    resolved = resolve(code) or resolve(label)
    if resolved:
        status = STATUS_OK if resolved in (code, label) else STATUS_CASE_CORRECTED
        return resolved, raw, status, confidence, all_intents

    return "Unclassified", raw, STATUS_INVALID_CATEGORY, confidence, all_intents


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

def predict_intent(
    model:         Llama,
    system_prompt: str,
    subject:       str,
    body:          str,
    valid_labels:  List[str],
    cfg:           Config,
    log:           logging.Logger,
) -> Tuple[str, str, str, Dict]:
    """
    Run the LLM on a single email.
    Returns: (label, raw_text, parse_status, prompt_info_dict)

    Uses official Mistral 7B Instruct v0.2 prompt format:
      <s>[INST] {system}\\n{subject}\\nBody:\\n{body} [/INST]
    Stop tokens include [/INST] to prevent runaway generation.
    """
    user_prompt = f"Subject: {subject}\nBody:\n{body}"
    full_prompt = f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]"
    prompt_tokens = len(model.tokenize(full_prompt.encode("utf-8")))

    if cfg.log_prompts or cfg.run_mode in ("dev", "test"):
        log_prompt("user", user_prompt, cfg, log)

    if prompt_tokens > cfg.n_ctx - OUTPUT_RESERVE_TOKENS:
        log.warning(
            "prompt_too_long",
            extra={
                "prompt_tokens": prompt_tokens,
                "n_ctx":         cfg.n_ctx,
                "subject_len":   len(subject),
                "body_len":      len(body),
            },
        )

    def _model_call():
        return model(
            full_prompt,
            max_tokens     = cfg.max_tokens,
            temperature    = cfg.temperature,
            top_p          = cfg.top_p,
            top_k          = cfg.top_k,
            repeat_penalty = cfg.repeat_penalty,
            stop           = ["</s>", "[INST]", "[/INST]", "\n\n\n"],
            echo           = False,
        )

    resp = call_with_retries(_model_call, cfg.infer_timeout_sec, cfg.infer_retries)

    raw_text = ""
    if isinstance(resp, dict):
        if "choices" in resp and resp["choices"]:
            c0 = resp["choices"][0]
            raw_text = c0.get("text", "") or c0.get("message", {}).get("content", "")
        elif "content" in resp:
            raw_text = resp["content"]
    elif isinstance(resp, str):
        raw_text = resp

    completion_tokens = len(model.tokenize(raw_text.encode("utf-8"))) if raw_text else 0
    total_tokens      = prompt_tokens + completion_tokens

    log.info(
        "token_usage",
        extra={
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
        },
    )

    if cfg.run_mode in ("dev", "test"):
        log_response(raw_text, cfg, log)

    label, _, parse_status, confidence, all_intents = parse_llm_output(raw_text, valid_labels)

    return label, raw_text, parse_status, {
        "system_prompt":     system_prompt,
        "user_prompt":       user_prompt,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "confidence":        confidence,
        "all_intents":       all_intents,
    }
