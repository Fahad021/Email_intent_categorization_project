#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hydro One Email Intent Classifier
FULL TELEMETRY VERSION (v4.7.0)
Reduced Knowledgebase + Human-Style Hybrid Prompt
No .env | All config via CLI arguments

Billing priority fixes (v4.7.0):
  - Keyword Hints section no longer says "supporting signals only" (contradicted billing rule)
  - Billing & Payments keywords now marked as MANDATORY triggers in hints header
  - Billing trigger condition expanded with more financial keywords
  - Third few-shot example added: implicit financial language → Billing & Payments

Mistral official prompting best practices applied (v4.6.0):
  - Markdown # headers replace -- SECTION -- plain text (model familiar from training)
  - Decision tree replaces numbered steps (eliminates priority contradictions)
  - Explicit tie-breaking rules replace blurry "strongest alignment" language
  - 2 few-shot examples added (recommended by docs for output format compliance)

[INST] prompt alignment (v4.5.0):
  - Single \\n between system prompt and email data (matches training distribution)
  - Email data goes directly after instruction — no redundant preamble
  - Format: <s>[INST] {system}\\n{subject}\\nBody:\\n{body} [/INST]

Mistral 7B Instruct v0.2 prompt alignment (v4.4.0):
  - <s> BOS token added (official requirement)
  - System prompt prepended inside [INST] (no native system role in v0.2)
  - add_bos=False in Llama() prevents double BOS
  - top_p 0.95 → 0.90, top_k 40 → 10 (tighter for classification)

GGUF / llama.cpp best practices (v4.3.0):
  - n_threads = physical cores via psutil
  - use_mlock optional: pins weights in RAM
  - flash_attn optional: faster on AVX2/AVX512
  - [/INST] in stop tokens: prevents runaway generation
  - use_mmap=True: fast model load

Prompt size management (v4.2.0):
  1. Dynamic body budget from actual system prompt token count
  2. System prompt budget check at startup
  3. Tiered body truncation: paragraph > sentence > hard cut
  4. Compact hints format
  5. KV-cache prefix reuse

Data integrity guarantees (v4.3.0):
  - llm_called flag per row
  - processing_status audit column per row
  - Row count reconciliation after pipeline
  - Skipped/error rows explicitly flagged
  - Integrity report in manifest
"""

from __future__ import annotations

import os
import re
import io
import sys
import json
import time
import hashlib
import textwrap
import logging
import argparse
import platform
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama

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
    parquet_in: str

    # Optional paths
    parquet_out: Optional[str] = None
    out_col:     str           = DEFAULT_OUT_COL
    record_dir:  Optional[str] = None
    log_file:    Optional[str] = None

    # LLM behaviour
    n_ctx:          int   = 8192
    n_gpu_layers:   int   = 0
    n_threads:      int   = field(default_factory=lambda: (
        psutil.cpu_count(logical=False) if _PSUTIL_AVAILABLE else (os.cpu_count() or 4)
    ))
    n_batch:        int   = 512
    max_tokens:     int   = 256
    temperature:    float = 0.05
    top_p:          float = 0.90   # tighter nucleus — reduces sampling pool for classification
    top_k:          int   = 10     # fewer candidates — more deterministic JSON output
    repeat_penalty: float = 1.1
    max_keywords:   int   = 5

    # GGUF best practices
    use_mlock:   bool = False   # pin model in RAM — set True if RAM >= 6 GB free
    flash_attn:  bool = False   # 20-40% faster on AVX2/AVX512 CPUs

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
        self.run_mode  = self.run_mode.lower()
        self.log_level = self.log_level.upper()
        if self.run_mode not in ("dev", "test", "prod"):
            raise ValueError(f"run_mode must be dev/test/prod, got: {self.run_mode}")


# --------------------------------------------------
# LOGGER
# --------------------------------------------------

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
                # integrity fields
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


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

PII_PATTERNS = [
    (re.compile(r"[A-Za-z0-9_.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<REDACTED_EMAIL>"),
    (re.compile(r"\b\d{9,16}\b"),                                     "<REDACTED_NUMBER>"),
    (re.compile(r"\b(?:\+?1\s?)?\(?\d{3}\)?[-.\s()]?\d{3}[-.\s]\d{4}\b"), "<REDACTED_PHONE>"),
]


def redact(text: str, enabled: bool = True, max_chars: int = 300) -> str:
    s = (text or "")[:max_chars]
    if not enabled:
        return s
    for pat, repl in PII_PATTERNS:
        s = pat.sub(repl, s)
    return s


def sha256_str(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def preview(text: str, max_chars: int = 300) -> str:
    return (text or "")[:max_chars]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------
# PROMPT BUDGET MANAGER
# --------------------------------------------------

class PromptBudget:
    """
    Calculated once at startup after model + system prompt are ready.
    Provides the exact body character limit per email based on
    actual system prompt token count — not a hardcoded guess.
    """
    SUBJECT_MAX = 300

    def __init__(self, model: Llama, system_prompt: str, cfg: Config, log: logging.Logger):
        self.system_tokens = len(model.tokenize(system_prompt.encode("utf-8")))
        self.n_ctx         = cfg.n_ctx

        safe_max = int(cfg.n_ctx * SYSTEM_PROMPT_MAX_RATIO)
        if self.system_tokens > safe_max:
            log.warning(
                "system_prompt_oversized",
                extra={
                    "system_tokens": self.system_tokens,
                    "safe_max":      safe_max,
                    "n_ctx":         cfg.n_ctx,
                    "pct_used":      round(self.system_tokens / cfg.n_ctx * 100, 1),
                },
            )
        else:
            log.info(
                "prompt_budget_ok",
                extra={
                    "system_tokens":   self.system_tokens,
                    "n_ctx":           cfg.n_ctx,
                    "pct_used":        round(self.system_tokens / cfg.n_ctx * 100, 1),
                    "body_char_limit": self.body_char_limit,
                },
            )

    @property
    def body_char_limit(self) -> int:
        available = (
            self.n_ctx
            - self.system_tokens
            - OUTPUT_RESERVE_TOKENS
            - OVERHEAD_TOKENS
            - int(self.SUBJECT_MAX / CHARS_PER_TOKEN)
        )
        return max(200, int(available * CHARS_PER_TOKEN))


# --------------------------------------------------
# KNOWLEDGEBASE LOADER
# --------------------------------------------------

def load_reduced_kb(
    path: str,
    log:  logging.Logger,
) -> Tuple[List[str], Dict[str, List[str]]]:
    df = pd.read_excel(path, sheet_name="Merged_Knowledgebase", engine="openpyxl")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Reduced_Category": "Category_Code", "Merged_Terms": "Term"})
    df = df[["Category_Code", "Term"]].dropna()

    valid_labels: List[str]              = sorted(df["Category_Code"].unique().tolist())
    category_terms: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        code  = str(row["Category_Code"]).strip()
        terms = [t.strip() for t in str(row["Term"]).strip().lower().split(",") if t.strip()]
        category_terms.setdefault(code, []).extend(terms)

    log.info("kb_loaded", extra={"file_in": path, "model": f"{len(valid_labels)} categories"})

    # Warn when cross-category term overlap is high — shared terms are less
    # discriminative and degrade classification quality, especially with many categories.
    term_counts = _compute_term_category_counts(category_terms)
    shared      = {t: c for t, c in term_counts.items() if c > 1}
    if shared:
        top_shared = sorted(shared.items(), key=lambda x: -x[1])[:10]
        overlap_pct = round(len(shared) / max(len(term_counts), 1) * 100, 1)
        log.warning(
            "kb_term_overlap_detected",
            extra={
                "shared_terms_count":  len(shared),
                "total_unique_terms":  len(term_counts),
                "overlap_pct":         overlap_pct,
                "top_shared_examples": [f"{t}({c})" for t, c in top_shared[:5]],
                "impact": (
                    "High overlap reduces prompt discriminativeness and may cause "
                    "misclassification. Consider consolidating shared terms into a "
                    "single canonical category or removing generic shared terms."
                ),
            },
        )

    return valid_labels, category_terms


# --------------------------------------------------
# TERM DISCRIMINATIVENESS HELPERS
# --------------------------------------------------

# Number of hint slots distributed across ALL categories before per-category
# scaling kicks in.  With max_keywords=5 this keeps total hints at ~50 terms
# for up to 10 categories, then scales down to preserve context-window budget.
KB_HINTS_TOTAL_BUDGET = 50


def _normalize_term(t: str) -> str:
    """Lowercase and strip a single term string."""
    return t.lower().strip()


def _compute_term_category_counts(
    category_terms: Dict[str, List[str]],
) -> Dict[str, int]:
    """
    Returns a mapping of each unique term (lowercased, stripped) to the number
    of categories it appears in.

    A term that appears in only one category is fully discriminative (count = 1).
    A term shared across many categories is less discriminative and is deprioritised
    in the prompt hints to reduce ambiguity for the LLM.

    Impact when the KB grows:
    - More categories → higher chance of common/generic terms appearing in multiple
      categories → more prompt noise → degraded classification accuracy.
    - This function makes that overlap visible so it can be acted on.
    """
    counts: Dict[str, int] = {}
    for terms in category_terms.values():
        seen_in_this_cat: set = set()
        for t in terms:
            t = _normalize_term(t)
            if t and t not in seen_in_this_cat:
                counts[t] = counts.get(t, 0) + 1
                seen_in_this_cat.add(t)
    return counts


def _rank_terms_by_discriminativeness(
    terms: List[str],
    term_category_counts: Dict[str, int],
) -> List[str]:
    """
    Returns *terms* deduplicated and sorted so the most category-exclusive
    (discriminative) terms appear first.

    Sort key (all ascending):
      1. Number of categories the term appears in — fewer is better (more unique).
      2. Word count — shorter phrases are cheaper in tokens.
      3. String length — shorter is cheaper in tokens.

    This ordering ensures that when ``max_keywords`` is applied the LLM sees
    the most distinguishing hints for every category, even when the KB is large.
    """
    deduped = list(dict.fromkeys(_normalize_term(t) for t in terms if _normalize_term(t)))
    return sorted(
        deduped,
        key=lambda t: (
            term_category_counts.get(t, 1),  # ascending: exclusive terms first
            len(t.split()),                  # ascending: fewer words first
            len(t),                          # ascending: shorter strings first
        ),
    )


# --------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------

def build_system_prompt(
    category_labels: List[str],
    category_terms:  Dict[str, List[str]],
    max_keywords:    int = 5,
) -> str:
    """
    Builds the system prompt aligned with Mistral official prompting best practices:

    Fix 1 — Markdown headers replace -- SECTION -- plain text.
             Markdown is explicitly recommended by docs as familiar to the model
             from training, readable, and parsable.

    Fix 2 — Decision tree replaces numbered steps with potential contradictions.
             Docs recommend if/otherwise branching to eliminate ambiguity.

    Fix 3 — "Strongest alignment" replaced with explicit tie-breaking rules.
             Docs say to avoid blurry words like "strongest", "better", "most".

    Fix 4 — Two few-shot examples added.
             Docs explicitly recommend examples to improve output format compliance,
             calling it the most effective tool for consistent JSON structure.
             Examples are placed AFTER the rules so the model sees the format
             instruction first, then sees it demonstrated.
    """

    # Fix 1: Markdown header for output format block
    output_block = textwrap.dedent("""
        # Output Format
        Return ONLY a single JSON object. No explanation. No preamble. No markdown.
        Required structure:
        {"intent_category":"<CAT>","intent_code":"<CAT>","confidence":"<LEVEL>","all_intents":["<CAT1>"]}
        - intent_category and intent_code MUST be identical and from Allowed Categories below
        - confidence: "high" | "medium" | "low"
        - all_intents: every plausible category detected (minimum one)
        - Your ENTIRE response must be this JSON object — nothing before or after it
    """).strip()

    # Fix 1: Markdown header for allowed categories
    allowed_block = "# Allowed Categories\n" + "\n".join(
        f"- {c}" for c in category_labels
    )

    # Fix 1: Markdown header for hints (compact format preserved for token efficiency)
    #
    # Two improvements for large knowledge bases:
    #
    # 1. Dynamic max_keywords scaling — with many categories the hints block grows
    #    linearly and can push the system prompt past SYSTEM_PROMPT_MAX_RATIO of the
    #    context window.  The per-category budget is capped so that the total number
    #    of hint terms stays close to KB_HINTS_TOTAL_BUDGET regardless of how many
    #    categories the KB contains.  Floor is 3 so every category retains at least
    #    a few discriminative signals.
    #
    # 2. Discriminativeness-first term ranking — sort terms so the ones unique to
    #    THIS category appear first, ahead of generic terms shared across many
    #    categories.  When max_keywords is applied, the LLM therefore sees the
    #    most distinguishing hints rather than the most common ones.
    n_cats = len(category_labels)
    per_cat_budget = KB_HINTS_TOTAL_BUDGET // max(n_cats, 1)
    effective_max  = max(3, min(max_keywords, per_cat_budget))

    term_category_counts = _compute_term_category_counts(category_terms)

    compact_hints = []
    for code in category_labels:
        terms = category_terms.get(code, [])
        if terms:
            ranked = _rank_terms_by_discriminativeness(terms, term_category_counts)
            compact_hints.append(f"- {code}: {', '.join(ranked[:effective_max])}")

    hints_block = (
        "# Keyword Hints\n"
        "Use these to identify likely categories.\n"
        "IMPORTANT: Billing & Payments keywords are MANDATORY triggers — when present,\n"
        "they override all other categories (see Classification Decision Tree below).\n"
        + "\n".join(compact_hints)
    )

    # Fix 2 + 3: Decision tree — billing priority moved to TOP (root cause fix).
    # Rule is now unconditional — no "among plausible" ambiguity.
    # Model reads billing override BEFORE any other reasoning branch.
    rules_block = textwrap.dedent("""
        # Classification Decision Tree
        Follow these branches in order. Stop at the first match.

        - IF the email is a delivery failure, bounce, mailer-daemon, or auto-reply:
            → intent_category = "Undeliverable", confidence = "high". STOP.

        - IF subject and body are both empty or contain only whitespace:
            → intent_category = "Unclassified", confidence = "low". STOP.

        - IF the email mentions anything related to bills, invoices, payments, charges,
          amounts owed, account balances, payment methods, rates, fees, late fees,
          overdue amounts, statements, billing cycles, refunds, credits, auto-pay,
          pre-authorized payments, or any financial transaction — even if other topics
          are also present:
            → intent_category = "Billing & Payments". ALWAYS. No exceptions. STOP.
            This rule overrides ALL other category matches without exception.

        - IF exactly one category matches the email meaning and keywords:
            → intent_category = that category. STOP.

        - IF multiple non-billing categories match:
            → Choose the category whose keywords appear in BOTH subject AND body.
            → If tied: choose the category with the most keyword matches in the body.
            → If still tied: choose the category whose first keyword appears earliest in the email.
            STOP.

        - IF no category fits with reasonable confidence:
            → intent_category = "Unclassified", confidence = "low". STOP.

        # Confidence Scale
        - high: subject AND body clearly match one category, multiple keyword hits
        - medium: reasonable match in body but subject is vague, or keywords partially match
        - low: weak signals only, conflicting intents, or forced best-guess
    """).strip()

    # Few-shot examples:
    # Example 1 deliberately shows MULTI-INTENT scenario where billing overrides another intent.
    # This is the exact failure case observed in production — billing detected but not selected.
    # Example 2 shows clean single-category match (outage).
    fewshot_block = textwrap.dedent("""
        # Examples

        ## Example 1 — Billing override (multi-intent email)
        Subject: I need to update my address and I also have a question about my bill
        Body:
        Hi, I recently moved and need to update my mailing address on my account. I also noticed my last bill had some charges I don't recognize. Can you help with both?
        Answer: {"intent_category":"Billing & Payments","intent_code":"Billing & Payments","confidence":"high","all_intents":["Billing & Payments","Move In / Move Out"]}

        ## Example 2 — Single intent (outage)
        Subject: Power has been out since last night on my street
        Body:
        Our entire street lost power around 10pm yesterday. I have a medical device that requires electricity. Please send a crew to restore power as soon as possible.
        Answer: {"intent_category":"Outage & Restoration","intent_code":"Outage & Restoration","confidence":"high","all_intents":["Outage & Restoration"]}

        ## Example 3 — Billing override (implicit financial language)
        Subject: Issue with my account
        Body:
        I checked my statement online and noticed the amount charged last month was higher than expected. I haven't changed my usage, so I'm not sure why my rate went up. Can someone look into this?
        Answer: {"intent_category":"Billing & Payments","intent_code":"Billing & Payments","confidence":"high","all_intents":["Billing & Payments"]}
    """).strip()

    header = (
        "You are an expert email intent classifier for Hydro One, a Canadian electric utility.\n"
        "Your task is to classify each customer email into exactly one intent category.\n"
        "Return valid JSON only — no explanation, no preamble, no markdown fences."
    )

    return "\n\n".join([
        header,
        output_block,
        allowed_block,
        hints_block,
        rules_block,
        fewshot_block,
    ]).strip()


# --------------------------------------------------
# TEXT HELPERS + TIERED TRUNCATION
# --------------------------------------------------

def _to_text(s: Optional[str]) -> str:
    return (s or "").replace("\r\n", "\n").strip()


def _collapse_ws(s: str) -> str:
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def smart_truncate_body(body: str, limit: int) -> str:
    """Tiered truncation: paragraph > sentence > hard cut."""
    if len(body) <= limit:
        return body
    cut = body[:limit]
    last_para = cut.rfind("\n\n")
    if last_para > int(limit * 0.6):
        return cut[:last_para].rstrip() + "…"
    last_sent = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if last_sent > int(limit * 0.6):
        return cut[:last_sent + 1].rstrip() + "…"
    return cut.rstrip() + "…"


def prepare_subject_body(subject: str, body: str, budget: PromptBudget) -> Tuple[str, str]:
    s = _to_text(subject)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = s[:budget.SUBJECT_MAX].rstrip()
    b = smart_truncate_body(_collapse_ws(_to_text(body)), budget.body_char_limit)
    return s, b


def build_user_prompt(subject: str, body: str) -> str:
    """
    Builds the user-facing portion of the prompt for telemetry logging.
    Matches the data section of the [INST] block exactly —
    no redundant preamble, data goes straight after the system instruction.
    """
    return f"Subject: {subject}\nBody:\n{body}"


# --------------------------------------------------
# MODEL LOADER  (GGUF best practices)
# --------------------------------------------------

def load_model(cfg: Config, log: logging.Logger) -> Llama:
    """
    GGUF best practices applied:
      - n_threads = physical cores (not logical) — passed via cfg
      - use_mmap  = True  (default) — fast load via memory mapping
      - use_mlock = cfg.use_mlock   — optional: pin weights in RAM
      - flash_attn= cfg.flash_attn  — optional: faster on AVX2/AVX512
      - last_n_tokens_size = n_ctx  — KV-cache prefix reuse
    """
    t0 = time.time()
    log.info(
        "model_load_start",
        extra={
            "file_in":     cfg.model_path,
            "n_threads":   cfg.n_threads,
            "use_mlock":   cfg.use_mlock,
            "flash_attn":  cfg.flash_attn,
            "n_ctx":       cfg.n_ctx,
        },
    )

    model = Llama(
        model_path         = cfg.model_path,
        n_ctx              = cfg.n_ctx,
        n_gpu_layers       = cfg.n_gpu_layers,
        n_threads          = cfg.n_threads,
        n_batch            = cfg.n_batch,
        last_n_tokens_size = cfg.n_ctx,    # KV-cache prefix reuse
        use_mmap           = True,         # fast load, uses half RAM
        use_mlock          = cfg.use_mlock,
        flash_attn         = cfg.flash_attn,
        add_bos            = False,
        verbose            = False,
    )

    log.info(
        "model_load_complete",
        extra={
            "model":        os.path.basename(cfg.model_path),
            "model_ctx":    cfg.n_ctx,
            "n_threads":    cfg.n_threads,
            "n_batch":      cfg.n_batch,
            "n_gpu_layers": cfg.n_gpu_layers,
            "use_mlock":    cfg.use_mlock,
            "flash_attn":   cfg.flash_attn,
            "latency_ms":   int((time.time() - t0) * 1000),
        },
    )
    return model


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
    Returns: label, raw, parse_status, confidence, all_intents
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
    Runs LLM for a single email.
    Returns: label, raw_text, parse_status, prompt_info

    GGUF stop tokens include [/INST] to prevent Mistral runaway generation.
    """
    # Official Mistral 7B Instruct v0.2 format from promptingguide.ai:
    #   <s>[INST] {instruction + data} [/INST]
    # - Single \n between system prompt and email data (not \n\n)
    # - Data goes directly after instruction — no redundant preamble
    # - <s> BOS added manually; add_bos=False in Llama() avoids double BOS
    # - Mistral v0.2 has no native system role; everything lives in [INST]
    user_prompt = (
        f"Subject: {subject}\n"
        f"Body:\n{body}"
    )
    full_prompt = (
        f"<s>[INST] {system_prompt}\n"
        f"{user_prompt} [/INST]"
    )
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
            # [/INST] added: prevents Mistral 7B runaway generation
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


# --------------------------------------------------
# TELEMETRY WRITERS
# --------------------------------------------------

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
    parquet_in:      str,
    parquet_out:     str,
    log_file:        Optional[str],
    predictions:     List[str],
    integrity_report: Dict,
    cfg:             Config,
    log:             logging.Logger,
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
        # Data integrity report embedded in manifest
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


# --------------------------------------------------
# PARQUET HELPERS
# --------------------------------------------------

def read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
    except Exception:
        df.to_parquet(path, engine="fastparquet", index=False)


# --------------------------------------------------
# DATA INTEGRITY CHECKER
# --------------------------------------------------

def build_integrity_report(
    df_in:              pd.DataFrame,
    predictions:        List[str],
    llm_called_flags:   List[bool],
    processing_statuses: List[str],
    parquet_out:        str,
    log:                logging.Logger,
) -> Dict:
    """
    Reconciles every row of input against every row of output.
    Logs a clear PASS or FAIL with full breakdown.
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

    # Core guarantee: every input row must have an output row
    row_count_match = rows_in == rows_out

    # Every non-skipped, non-errored row must have had LLM called
    expected_llm_calls = rows_in - rows_skipped
    llm_call_match     = rows_llm_called == expected_llm_calls

    passed = row_count_match and llm_call_match

    report = {
        "passed":             passed,
        "rows_in":            rows_in,
        "rows_out":           rows_out,
        "row_count_match":    row_count_match,
        "rows_llm_called":    rows_llm_called,
        "expected_llm_calls": expected_llm_calls,
        "llm_call_match":     llm_call_match,
        "rows_ok":            rows_ok,
        "rows_case_corrected": sum(1 for s in processing_statuses if s == STATUS_CASE_CORRECTED),
        "rows_skipped_empty": rows_skipped,
        "rows_errored":       rows_errored,
        "rows_parse_issues":  rows_parse_issue,
        "out_file":           parquet_out,
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
            issues.append(
                f"LLM CALL MISMATCH: expected={expected_llm_calls} actual={rows_llm_called}"
            )
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


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run(cfg: Config) -> None:
    """
    Full pipeline:
      1.  Build logger
      2.  Load KB + build system prompt
      3.  Load model (GGUF best practices)
      4.  Compute prompt budget
      5.  Read & validate input parquet
      6.  Inference loop — every row explicitly tracked
      7.  Data integrity reconciliation
      8.  Write output parquet (with audit columns)
      9.  Write manifest (with integrity report)
    """
    log = build_logger(cfg)

    # KB + prompt
    valid_labels, category_terms = load_reduced_kb(cfg.kb_file, log)
    system_prompt = build_system_prompt(valid_labels, category_terms, cfg.max_keywords)
    log.info("system_prompt_built", extra={"prompt_chars": len(system_prompt)})
    if cfg.log_prompts or cfg.run_mode in ("dev", "test"):
        log_prompt("system", system_prompt, cfg, log)

    # Model
    model = load_model(cfg, log)

    # Prompt budget
    budget = PromptBudget(model, system_prompt, cfg, log)
    log.info(
        "body_budget_set",
        extra={
            "body_char_limit": budget.body_char_limit,
            "system_tokens":   budget.system_tokens,
        },
    )

    # Input
    df = read_parquet(cfg.parquet_in)
    missing = [c for c in ("GUID", "Subject", "Body") if c not in df.columns]
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")

    rows_in = len(df)
    log.info("input_loaded", extra={"file_in": cfg.parquet_in, "rows_in": rows_in})

    # Output path
    parquet_out = cfg.parquet_out
    if not parquet_out:
        stem = os.path.splitext(os.path.basename(cfg.parquet_in))[0]
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_out = os.path.join(
            os.path.dirname(cfg.parquet_in),
            f"{stem}_with_llm_predictions_{ts}.parquet",
        )

    # Telemetry dirs
    records_base = cfg.record_dir or os.path.join(
        os.path.dirname(parquet_out), "inference_records"
    )
    prompts_dir = os.path.join(records_base, "prompts")
    if not cfg.no_records:
        ensure_dir(records_base)
        ensure_dir(prompts_dir)

    # Per-row output columns
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
        t0 = time.time()

        # ---- PATH A: empty input — skip LLM, flag explicitly ----
        if not subject and not body:
            label              = "Unclassified"
            raw                = ""
            parse_stat         = STATUS_SKIPPED_EMPTY
            processing_status  = STATUS_SKIPPED_EMPTY
            llm_called         = False
            error_detail       = ""
            prompt_info: Dict  = {
                "system_prompt":     system_prompt,
                "user_prompt":       "",
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "total_tokens":      0,
                "confidence":        "low",
                "all_intents":       [],
            }

        # ---- PATH B: normal — call LLM ----
        else:
            llm_called   = True
            error_detail = ""
            try:
                label, raw, parse_stat, prompt_info = predict_intent(
                    model=model, system_prompt=system_prompt,
                    subject=subject, body=body,
                    valid_labels=valid_labels, cfg=cfg, log=log,
                )
                # processing_status mirrors parse_stat for successful LLM calls
                processing_status = parse_stat

            # ---- PATH C: exception — LLM was attempted, store error detail ----
            except Exception as e:
                label             = "Unclassified"
                raw               = ""
                parse_stat        = STATUS_ERROR
                processing_status = STATUS_ERROR
                error_detail      = str(e)
                prompt_info = {
                    "system_prompt":     system_prompt,
                    "user_prompt":       "",
                    "prompt_tokens":     0,
                    "completion_tokens": 0,
                    "total_tokens":      0,
                    "confidence":        "low",
                    "all_intents":       [],
                }
                log.error(
                    "inference_error",
                    extra={
                        "guid":       guid,
                        "row_index":  idx,
                        "error":      error_detail,
                    },
                )

        latency_ms = int((time.time() - t0) * 1000)

        predictions.append(label)
        parse_statuses.append(parse_stat)
        processing_statuses.append(processing_status)
        llm_called_flags.append(llm_called)
        confidences.append(prompt_info.get("confidence", "unknown"))
        all_intents_col.append(json.dumps(prompt_info.get("all_intents", [])))
        error_details.append(error_detail)

        log.info(
            "inference_done",
            extra={
                "guid":              guid,
                "row_index":         idx,
                "latency_ms":        latency_ms,
                "intent_code":       label,
                "intent":            label,
                "subject_len":       len(subject),
                "body_len":          len(body),
                "body_budget":       budget.body_char_limit,
                "response_raw":      redact(raw, enabled=cfg.redact_logs, max_chars=500),
                "parse_status":      parse_stat,
                "processing_status": processing_status,
                "llm_called":        llm_called,
                "confidence":        prompt_info.get("confidence", "unknown"),
                "file_in":           cfg.parquet_in,
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
            try:
                write_inference_record(
                    out_dir=records_base,         guid=guid,
                    row_index=idx,                subject=subject,
                    body=body,
                    user_prompt=prompt_info["user_prompt"],
                    system_prompt=prompt_info["system_prompt"],
                    raw_response=raw,             label=label,
                    parse_status=parse_stat,
                    processing_status=processing_status,
                    llm_called=llm_called,
                    latency_ms=latency_ms,        parquet_in=cfg.parquet_in,
                    prompt_tokens=prompt_info["prompt_tokens"],
                    completion_tokens=prompt_info["completion_tokens"],
                    total_tokens=prompt_info["total_tokens"],
                    confidence=prompt_info.get("confidence", "unknown"),
                    all_intents=prompt_info.get("all_intents", []),
                    body_budget=budget.body_char_limit,
                    cfg=cfg,
                )
                if llm_called:
                    write_prompt_record(
                        out_dir=prompts_dir,      row_index=idx,
                        guid=guid,
                        system_prompt=prompt_info["system_prompt"],
                        user_prompt=prompt_info["user_prompt"],
                        prompt_tokens=prompt_info["prompt_tokens"],
                        completion_tokens=prompt_info["completion_tokens"],
                        total_tokens=prompt_info["total_tokens"],
                        cfg=cfg,
                    )
            except Exception as e:
                log.warning(
                    "record_write_failed",
                    extra={"guid": guid, "row_index": idx, "msg": str(e)},
                )

    # ---- DATA INTEGRITY CHECK ----
    integrity_report = build_integrity_report(
        df_in=df,
        predictions=predictions,
        llm_called_flags=llm_called_flags,
        processing_statuses=processing_statuses,
        parquet_out=parquet_out,
        log=log,
    )

    # ---- WRITE OUTPUT ----
    # All audit columns written alongside prediction
    df[cfg.out_col]          = predictions
    df["Parse_Status"]        = parse_statuses
    df["Processing_Status"]   = processing_statuses   # full audit trail
    df["LLM_Called"]          = llm_called_flags       # was LLM actually invoked?
    df["Confidence"]          = confidences
    df["All_Intents"]         = all_intents_col
    df["Error_Detail"]        = error_details          # populated only on exceptions

    write_parquet(df, parquet_out)

    log.info(
        "output_written",
        extra={
            "file_out":   parquet_out,
            "rows_total": len(df),
        },
    )

    write_manifest(
        parquet_in=cfg.parquet_in,
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
            "rows_total":      len(df),
            "duration_s":      round(time.time() - t_run_start, 3),
            "file_in":         cfg.parquet_in,
            "file_out":        parquet_out,
            "integrity_passed": integrity_report["passed"],
        },
    )

    # Raise if integrity failed so CI/schedulers can catch it
    if not integrity_report["passed"]:
        raise RuntimeError(
            f"Data integrity check FAILED. See manifest at {parquet_out}.manifest.json"
        )


# --------------------------------------------------
# CLI
# --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hydro One Email Intent Classifier v4.6.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    req = p.add_argument_group("Required")
    req.add_argument("--in",    dest="parquet_in", required=True,
                     help="Input Parquet file (must contain GUID, Subject, Body).")
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
                     default=(psutil.cpu_count(logical=False) if _PSUTIL_AVAILABLE else (os.cpu_count() or 4)),
                     help="Physical CPU cores. Do not use logical/hyperthreaded count.")
    llm.add_argument("--n-batch",        dest="n_batch",        type=int,   default=512)
    llm.add_argument("--max-tokens",     dest="max_tokens",     type=int,   default=256)
    llm.add_argument("--temperature",    dest="temperature",    type=float, default=0.05)
    llm.add_argument("--top-p",          dest="top_p",          type=float, default=0.90,
                     help="Nucleus sampling. 0.90 recommended for classification (tighter than creative tasks).")
    llm.add_argument("--top-k",          dest="top_k",          type=int,   default=10,
                     help="Top-k sampling. 10 recommended for classification (more deterministic JSON output).")
    llm.add_argument("--repeat-penalty", dest="repeat_penalty", type=float, default=1.1)
    llm.add_argument("--max-keywords",   dest="max_keywords",   type=int,   default=5)

    gguf = p.add_argument_group("GGUF Best Practices")
    gguf.add_argument("--use-mlock",  dest="use_mlock",  action="store_true",
                      help="Pin model weights in RAM (prevents swap during batch run). Requires free RAM >= model size.")
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
        logs_dir = os.path.join(
            os.path.dirname(args.parquet_out or args.parquet_in), "logs"
        )
        ensure_dir(logs_dir)
        log_file = os.path.join(
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
