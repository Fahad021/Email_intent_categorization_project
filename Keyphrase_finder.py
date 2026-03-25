"""
Email Keyphrase Matching Script
================================
For EACH email, searches against EVERY category's terms.
Output: ONE new column 'keyphrase_stats' containing a JSON object like:

{
    "Account_Details":   {"account history": 2, "legacy": 1},
    "Account_Inquiries": {"balance inquiry": 1},
    "Billing":           {}          ← empty dict = no match
}

Usage:
    python email_keyphrase_matching.py \
        --input  emails.parquet \
        --json   merged_by_category.json \
        --output emails_with_keyphrases.parquet
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import unicodedata


# ══════════════════════════════════════════════════════════════
# 1.  TEXT CLEANING
# ══════════════════════════════════════════════════════════════

def remove_email_headers(text: str) -> str:
    for p in [r"From:.*", r"To:.*", r"Cc:.*", r"Bcc:.*",
              r"Sent:.*", r"Subject:.*", r"Date:.*",
              r"-{2,}.*Original Message.*-{2,}", r"_{3,}"]:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text


def remove_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    for entity, rep in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")]:
        text = text.replace(entity, rep)
    return text


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    text = remove_html(text)
    text = remove_email_headers(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)             # email addresses
    text = re.sub(r"[^a-zA-Z\s]", " ", text)         # keep letters only
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════
# 2.  LOAD CATEGORY TERMS FROM JSON
# ══════════════════════════════════════════════════════════════

def load_categories(json_path: str) -> dict:
    """
    Returns:
    {
        "Account_Details":  ["heritage data library hdl", "account history", ...],  # longest first
        "Account_Inquiries": ["balance inquiry", ...],
        ...
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {}
    for code, info in data.items():
        raw_terms = info.get("terms", [])
        cleaned_terms = sorted(
            {t.strip().lower() for t in raw_terms if t.strip()},
            key=len,
            reverse=True    # longest first so multi-word phrases match before sub-words
        )
        categories[code] = cleaned_terms

    print(f"  -> {len(categories)} categories loaded:")
    for code, terms in categories.items():
        print(f"       {code:<35}  ({len(terms)} terms)")

    return categories


# ══════════════════════════════════════════════════════════════
# 3.  BUILD keyphrase_stats FOR ONE EMAIL
# ══════════════════════════════════════════════════════════════

def build_keyphrase_stats(text: str, categories: dict) -> str:
    """
    For one email's combined text, match every category's terms.

    Returns a JSON string:
    {
        "Account_Details":   {"account history": 2, "legacy": 1},
        "Account_Inquiries": {},
        ...
    }
    """
    result = {}

    for code, terms in categories.items():
        counts = {}
        for term in terms:
            hits = re.findall(r"\b" + re.escape(term) + r"\b", text)
            if hits:
                counts[term] = len(hits)
        if counts:                  # only include category if at least one term matched
            result[code] = counts

    return json.dumps(result, ensure_ascii=False) if result else None


# ══════════════════════════════════════════════════════════════
# 4.  SUMMARY
# ══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, categories: dict):
    total = len(df)
    print("\n" + "=" * 60)
    print("  MATCH SUMMARY  (emails with >= 1 term hit per category)")
    print("=" * 60)
    print(f"  {'Category Code':<35} {'Matched':>8}  {'%':>6}")
    print("-" * 60)

    # Parse the JSON column once
    parsed = df["keyphrase_stats"].apply(json.loads)

    for code in categories:
        n = parsed.apply(lambda d: bool(d.get(code))).sum()
        pct = 100 * n / total if total else 0
        print(f"  {code:<35} {n:>8,}  {pct:>5.1f}%")

    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════

def process_emails(input_path: str, json_path: str, output_path: str = None):
    input_path = Path(input_path)

    # ── Load input ─────────────────────────────────────────────
    print(f"\n[1/5] Loading input: {input_path}")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")
    print(f"       {len(df):,} rows  |  columns: {list(df.columns)}")

    # ── Detect Subject / Body columns ──────────────────────────
    print("\n[2/5] Detecting Subject / Body columns...")
    subject_col = next(
        (c for c in df.columns if re.sub(r"[\[\]]", "", c).strip().lower() == "subject"), None
    )
    body_col = next(
        (c for c in df.columns if re.sub(r"[\[\]]", "", c).strip().lower() == "body"), None
    )
    if not subject_col or not body_col:
        raise ValueError(f"Could not find Subject/Body. Available: {list(df.columns)}")
    print(f"       Subject -> '{subject_col}'   |   Body -> '{body_col}'")

    # ── Clean text ─────────────────────────────────────────────
    print("\n[3/5] Cleaning text...")
    clean_subject = df[subject_col].apply(clean_text)
    clean_body    = df[body_col].apply(clean_text)
    # Subject weighted 2x (more signal-dense)
    combined      = (clean_subject + " ") * 2 + clean_body

    # ── Load categories ────────────────────────────────────────
    print(f"\n[4/5] Loading categories from: {json_path}")
    categories = load_categories(json_path)

    # ── Build keyphrase_stats column ───────────────────────────
    print("\n[5/5] Building keyphrase_stats column...")
    df["keyphrase_stats"] = combined.apply(
        lambda text: build_keyphrase_stats(text, categories)
    )

    # ── Summary ────────────────────────────────────────────────
    print_summary(df, categories)

    # ── Save ───────────────────────────────────────────────────
    if output_path is None:
        output_path = input_path.parent / (input_path.stem + "_with_keyphrases.parquet")
    output_path = Path(output_path)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"\n[OK]  Saved -> {output_path}")

    # ── Quick peek ─────────────────────────────────────────────
    print("\nSample keyphrase_stats (first 3 rows):")
    for i, val in enumerate(df["keyphrase_stats"].head(3)):
        parsed = json.loads(val)
        # Show only categories that had at least one hit
        hits = {k: v for k, v in parsed.items() if v}
        print(f"\n  Row {i}: {json.dumps(hits, indent=4)}")

    return df


# ══════════════════════════════════════════════════════════════
# 6.  CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email Keyphrase Matching - single JSON column")
    parser.add_argument("--input",  required=True, help="Input .parquet or .csv")
    parser.add_argument("--json",   required=True, help="merged_by_category.json")
    parser.add_argument("--output", default=None,  help="Output file path (optional)")
    args = parser.parse_args()

    process_emails(
        input_path=args.input,
        json_path=args.json,
        output_path=args.output,
    )
