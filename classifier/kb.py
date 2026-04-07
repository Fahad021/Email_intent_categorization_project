from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd


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
    return valid_labels, category_terms
