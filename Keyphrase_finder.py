"""Email Keyphrase Matching Script

- Reads a parquet or CSV file containing email data
- Cleans email Subject and Body text
- Matches terms from merged_by_category.json
- Adds a 'key_phrase' column with matched terms and their occurrence counts
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# ─────────────────────────────────────────────
# 1. TEXT CLEANING
# ─────────────────────────────────────────────

def remove_email_headers(text: str) -> str:
    """Strip common forwarded/reply header lines."""
    patterns = [
        r"^From:.*$",
        r"^To:.*$",
        r"^Cc:.*$",
        r"^Sent:.*$",
        r"^Subject:.*$",
        r"^Date:.*$",
        r"-{2,}.*Original Message.*-{2,}",
        r"_{2,}",
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text

def remove_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize common HTML entities / non-breaking spaces
    text = text.replace("\u00a0", " ")  # nbsp
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    return text

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to closest ASCII equivalent."""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)

def remove_email_addresses(text: str) -> str:
    return re.sub(r"\S+@\S+", " ", text)

def remove_punctuation_and_numbers(text: str) -> str:
    # Keep letters and spaces only
    return re.sub(r"[^a-zA-Z\s]", " ", text)

def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: object) -> str:
    """Full cleaning pipeline."""
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = remove_html(text)
    text = remove_email_headers(text)
    text = remove_urls(text)
    text = remove_email_addresses(text)
    text = remove_punctuation_and_numbers(text)
    text = text.lower()
    text = collapse_whitespace(text)

    return text

# ─────────────────────────────────────────────
# 2. LOAD CATEGORY TERMS
# ─────────────────────────────────────────────

def load_terms(json_path: str | Path) -> Tuple[dict[str, str], list[str]]:
    """Load merged_by_category.json.

    Returns:
      - term_to_category: {term_lower: category_label}
      - sorted_terms: unique terms sorted longest-first (prefer multi-word matches)
    """
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    term_to_category: dict[str, str] = {}
    for category_key, category_data in data.items():
        label = category_data.get("category_label", category_key)
        for term in category_data.get("terms", []):
            term_clean = str(term).strip().lower()
            if term_clean:
                term_to_category[term_clean] = label

    sorted_terms = sorted(term_to_category.keys(), key=len, reverse=True)
    return term_to_category, sorted_terms

# ─────────────────────────────────────────────
# 3. KEYPHRASE MATCHING
# ─────────────────────────────────────────────

def find_keyphrases(cleaned_text: str, sorted_terms: list[str]) -> Optional[str]:
    """Find all term matches in cleaned_text.

    Returns a readable string: "term1(2), term2(1)" or None if no matches.
    Uses whole-word boundary matching.
    """
    if not cleaned_text:
        return None

    matches: Counter[str] = Counter()

    for term in sorted_terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        found = re.findall(pattern, cleaned_text)
        if found:
            matches[term] += len(found)

    if not matches:
        return None

    return ", ".join(f"{term}({count})" for term, count in matches.most_common())

# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────

def process_emails(input_path: str | Path, json_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    input_path = Path(input_path)
    json_path = Path(json_path)

    # ── Load data ──────────────────────────────
    print(f"Loading input file: {input_path}")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in (".csv", ".tsv"):
        # use python engine for .tsv if separator differs; keep default csv for simplicity
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    print(f"  → {len(df):,} rows loaded")
    print(f"  → Columns: {list(df.columns)}")

    # ── Load terms ─────────────────────────────
    print(f"\nLoading category terms from: {json_path}")
    _term_to_category, sorted_terms = load_terms(json_path)
    print(f"  → {len(sorted_terms)} unique terms loaded")

    # ── Clean text ─────────────────────────────
    print("\nCleaning Subject and Body columns...")
    subject_col = next((c for c in df.columns if c.lower() in ("subject", "[subject]")), None)
    body_col = next((c for c in df.columns if c.lower() in ("body", "[body]")), None)

    if subject_col is None or body_col is None:
        raise ValueError(f"Could not find Subject/Body columns. Found: {list(df.columns)}")

    df["cleaned_subject"] = df[subject_col].apply(clean_text)
    df["cleaned_body"] = df[body_col].apply(clean_text)

    # Combine subject + body for matching (subject weighted 2x)
    df["combined_text"] = df["cleaned_subject"] + " " + df["cleaned_subject"] + " " + df["cleaned_body"]

    # ── Match keyphrases ───────────────────────
    print("Matching keyphrases...")
    df["key_phrase"] = df["combined_text"].apply(lambda text: find_keyphrases(text, sorted_terms))

    matched = df["key_phrase"].notna().sum()
    print(f"  → {matched:,} / {len(df):,} emails matched at least one term")

    # ── Drop helper columns ────────────────────
    df.drop(columns=["combined_text"], inplace=True)

    # ── Save output ────────────────────────────
    if output_path is None:
        output_path = input_path.parent / (input_path.stem + "_with_keyphrases.parquet")
    output_path = Path(output_path)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"\nOutput saved to: {output_path}")
    print("\nSample results:")
    print(df[["cleaned_subject", "key_phrase"]].head(10).to_string(index=False))

    return df

# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Email Keyphrase Matching")
    parser.add_argument("--input", required=True, help="Path to input .parquet or .csv file")
    parser.add_argument("--json", required=True, help="Path to merged_by_category.json")
    parser.add_argument("--output", default=None, help="Path for output file (optional)")
    args = parser.parse_args()

    process_emails(
        input_path=args.input,
        json_path=args.json,
        output_path=args.output,
    )

if __name__ == "__main__":
    main()