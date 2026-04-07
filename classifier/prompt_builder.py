from __future__ import annotations

import logging
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

from llama_cpp import Llama

from .config import (
    Config,
    CHARS_PER_TOKEN,
    OVERHEAD_TOKENS,
    OUTPUT_RESERVE_TOKENS,
    SYSTEM_PROMPT_MAX_RATIO,
)


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
# PROMPT BUILDER
# --------------------------------------------------

def _render_allowed_categories(category_labels: List[str]) -> str:
    return "\n".join(f"- {c}" for c in category_labels)


def _render_keyword_hints(
    category_labels: List[str],
    category_terms:  Dict[str, List[str]],
    max_keywords:    int,
) -> str:
    lines = []
    for code in category_labels:
        terms = category_terms.get(code, [])
        if terms:
            deduped = list(dict.fromkeys(t.lower().strip() for t in terms if t.strip()))
            deduped_sorted = sorted(deduped, key=lambda t: (len(t.split()), len(t)))
            lines.append(f"- {code}: {', '.join(deduped_sorted[:max_keywords])}")
    return "\n".join(lines)


def build_system_prompt(
    category_labels: List[str],
    category_terms:  Dict[str, List[str]],
    max_keywords:    int = 5,
    prompt_file:     str = "",
) -> str:
    """
    Builds the system prompt.

    When prompt_file is set, loads the template from disk and fills two placeholders:
      {{ALLOWED_CATEGORIES}}  — "- Category Name" lines (one per category)
      {{KEYWORD_HINTS}}       — "- Category: kw1, kw2, ..." lines (from KB)

    When prompt_file is empty (default), uses the built-in hardcoded prompt.
    """
    if prompt_file:
        with open(prompt_file, encoding="utf-8") as fh:
            template = fh.read()
        return template.replace(
            "{{ALLOWED_CATEGORIES}}", _render_allowed_categories(category_labels)
        ).replace(
            "{{KEYWORD_HINTS}}", _render_keyword_hints(category_labels, category_terms, max_keywords)
        ).strip()

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

    allowed_block = "# Allowed Categories\n" + _render_allowed_categories(category_labels)

    hints_block = (
        "# Keyword Hints\n"
        "Use these to identify likely categories.\n"
        "IMPORTANT: Billing & Payments keywords are MANDATORY triggers — when present,\n"
        "they override all other categories (see Classification Decision Tree below).\n"
        + _render_keyword_hints(category_labels, category_terms, max_keywords)
    )

    rules_block = textwrap.dedent("""
        # Classification Decision Tree
        Follow these branches in order. Stop at the first match.

        - IF the email is a delivery failure, bounce, mailer-daemon, or auto-reply:
            -> intent_category = "Undeliverable", confidence = "high". STOP.

        - IF subject and body are both empty or contain only whitespace:
            -> intent_category = "Unclassified", confidence = "low". STOP.

        - IF the email mentions anything related to bills, invoices, payments, charges,
          amounts owed, account balances, payment methods, rates, fees, late fees,
          overdue amounts, statements, billing cycles, refunds, credits, auto-pay,
          pre-authorized payments, or any financial transaction — even if other topics
          are also present:
            -> intent_category = "Billing & Payments". ALWAYS. No exceptions. STOP.
            This rule overrides ALL other category matches without exception.

        - IF exactly one category matches the email meaning and keywords:
            -> intent_category = that category. STOP.

        - IF multiple non-billing categories match:
            -> Choose the category whose keywords appear in BOTH subject AND body.
            -> If tied: choose the category with the most keyword matches in the body.
            -> If still tied: choose the category whose first keyword appears earliest in the email.
            STOP.

        - IF no category fits with reasonable confidence:
            -> intent_category = "Unclassified", confidence = "low". STOP.

        # Confidence Scale
        - high: subject AND body clearly match one category, multiple keyword hits
        - medium: reasonable match in body but subject is vague, or keywords partially match
        - low: weak signals only, conflicting intents, or forced best-guess
    """).strip()

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

    return "\n\n".join([header, output_block, allowed_block, hints_block, rules_block, fewshot_block]).strip()


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
        return cut[:last_para].rstrip() + "..."
    last_sent = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if last_sent > int(limit * 0.6):
        return cut[:last_sent + 1].rstrip() + "..."
    return cut.rstrip() + "..."


def prepare_subject_body(subject: str, body: str, budget: PromptBudget) -> Tuple[str, str]:
    s = _to_text(subject)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = s[:budget.SUBJECT_MAX].rstrip()
    b = smart_truncate_body(_collapse_ws(_to_text(body)), budget.body_char_limit)
    return s, b


def build_user_prompt(subject: str, body: str) -> str:
    """Builds the user-facing portion of the prompt. Useful for external callers."""
    return f"Subject: {subject}\nBody:\n{body}"
