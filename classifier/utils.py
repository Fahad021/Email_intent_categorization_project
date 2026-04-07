from __future__ import annotations

import hashlib
import os
import re
from typing import List, Tuple

PII_PATTERNS: List[Tuple] = [
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
