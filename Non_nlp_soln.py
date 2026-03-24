# “””
Three-Stage Deterministic Intent Pipeline

No LLM needed for the core logic. Pure rule-based retrieval.

Stage 1 — KEYWORD MATCH
Scan email Body + Subject for category terms (keywords).
Each keyword hit maps to a Category_Code.
Result: set of matched Category_Code(s)

Stage 2 — INTENT LOOKUP
For each matched Category_Code, look up all possible Intent_Codes.
(Category_Code == Intent_Code in your schema, but one category
can have multiple intent variants e.g. Billing, Billing_BS, Billing_DM)
Result: candidate Intent_Code(s)

Stage 3 — EMAIL PRUNE
Each Intent_Code has an associated email address (the “To” field
in your intent file). Match this against the actual email’s “To”
address to prune candidates.
Result: final confirmed Intent_Code(s)

LLM is used ONLY as a fallback when the deterministic pipeline
produces no result (zero keyword matches or no email match).

Data relationships used:
category file : Category_Code → [term1, term2, …]   (hundreds of terms)
intent file   : Intent_Code   → Email (To address)
Intent_Code   → Intent label

Requirements:
pip install pandas openpyxl pyodbc sqlalchemy tqdm
pip install llama-cpp-python langchain langchain-community  # fallback only
“””

# ─────────────────────────────────────────────────────────────

# CONFIGURATION

# ─────────────────────────────────────────────────────────────

CATEGORY_FILE   = r”C:\data\SPSS_Category_Concepts.xlsx”
INTENT_FILE     = r”C:\data\Email_Intents.xlsx”

SQL_SERVER      = “GENESYSCICDBQA.corp.hyd…”
SQL_DATABASE    = “EMAIL”
SQL_TABLE       = “[dbo].[Original_Email]”
SQL_TRUSTED     = True

BATCH_SIZE      = 100           # can be large now — pipeline is fast
DRY_RUN         = False

# Keyword matching

MIN_KEYWORD_LENGTH  = 3         # ignore very short terms
KEYWORD_SCORE_MODE  = “count”   # “count” | “weighted” (longer terms score higher)

# Email matching

EMAIL_MATCH_MODE    = “exact”   # “exact” | “domain” | “fuzzy”
# exact  = full address must match
# domain = just the domain must match
# fuzzy  = substring match

# Fallback LLM (only used when pipeline returns nothing)

USE_LLM_FALLBACK    = True
GGUF_MODEL_PATH     = r”C:\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf”
N_GPU_LAYERS        = 0
N_CTX               = 4096

# ─────────────────────────────────────────────────────────────

import re
import logging
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s [%(levelname)s] %(message)s”,
datefmt=”%H:%M:%S”,
)
log = logging.getLogger(**name**)

# ══════════════════════════════════════════════════════════════

# DATA STRUCTURES

# ══════════════════════════════════════════════════════════════

@dataclass
class IntentRecord:
“”“One row from the intent file.”””
intent_code:  str
intent_label: str
email:        str           # the “To” address this intent is associated with
method:       str = “”

@dataclass
class PipelineResult:
“”“Result for one email.”””
guid:              str
matched_categories: list[str]     = field(default_factory=list)
candidate_intents:  list[str]     = field(default_factory=list)
final_intents:      list[str]     = field(default_factory=list)
match_source:       str           = “”   # “keyword+email” | “keyword_only” | “llm_fallback”
keyword_hits:       dict          = field(default_factory=dict)
confidence:         str           = “high”

# ══════════════════════════════════════════════════════════════

# 1. LOAD & INDEX REFERENCE FILES

# ══════════════════════════════════════════════════════════════

class IntentKnowledgeBase:
“””
Holds all reference data and provides fast lookup methods.

```
Internal indexes built at startup:
  term_index        : {term_string → [Category_Code, ...]}
  category_intents  : {Category_Code → [IntentRecord, ...]}
  email_intents     : {email_address → [IntentRecord, ...]}
  intent_by_code    : {intent_code → IntentRecord}
"""

def __init__(self, category_path: str, intent_path: str):
    self._load_category_file(category_path)
    self._load_intent_file(intent_path)
    self._build_indexes()
    self._log_summary()

def _load_category_file(self, path: str):
    """
    Loads:  Category_Code → [term1, term2, ...]
    From:   Excel columns [Category, Category_Code, Term]
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = df.columns.str.strip()

    key_col  = "Category_Code" if "Category_Code" in df.columns else "Category"
    term_col = "Term" if "Term" in df.columns else df.columns[2]

    df = df[[key_col, term_col]].dropna()

    self.category_terms: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        code = str(row[key_col]).strip()
        term = str(row[term_col]).strip().lower()
        if len(term) >= MIN_KEYWORD_LENGTH:
            self.category_terms.setdefault(code, []).append(term)

    log.info(
        "Category file: %d codes, %d total terms",
        len(self.category_terms),
        sum(len(t) for t in self.category_terms.values())
    )

def _load_intent_file(self, path: str):
    """
    Loads:  Intent_Code → email address (the "To" for that intent)
    From:   Excel columns [ID, Email, Intent, Intent_Code, Method]

    NOTE: In your data, the "Email" column in the intent file
          is the DESTINATION email address — i.e. which inbox
          this intent routes to. This is the prune key in Stage 3.
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = df.columns.str.strip()

    # Normalise column names
    email_col  = "Email"
    intent_col = "Intent"
    code_col   = "Intent_Code"
    method_col = "Method" if "Method" in df.columns else None

    df = df.dropna(subset=[code_col, email_col])
    df[code_col]   = df[code_col].str.strip()
    df[intent_col] = df[intent_col].str.strip()
    df[email_col]  = df[email_col].str.strip().str.lower()

    self.intent_records: list[IntentRecord] = []
    for _, row in df.iterrows():
        self.intent_records.append(IntentRecord(
            intent_code  = str(row[code_col]),
            intent_label = str(row[intent_col]),
            email        = str(row[email_col]),
            method       = str(row[method_col]) if method_col else "",
        ))

    log.info(
        "Intent file: %d records, %d unique codes, %d unique email addresses",
        len(self.intent_records),
        len({r.intent_code for r in self.intent_records}),
        len({r.email for r in self.intent_records}),
    )

def _build_indexes(self):
    """Build all lookup indexes from loaded data."""

    # ── Index 1: term → [Category_Code] ───────────────────
    # Allows O(1) lookup: given a word, which categories match?
    self.term_index: dict[str, list[str]] = defaultdict(list)
    for code, terms in self.category_terms.items():
        for term in terms:
            self.term_index[term.lower()].append(code)

    # Also build compiled regex patterns for multi-word terms
    # Single-word terms use dict lookup (faster)
    # Multi-word terms need regex for boundary matching
    self.multiword_patterns: list[tuple[re.Pattern, list[str]]] = []
    self.singleword_terms:   dict[str, list[str]] = {}

    for term, codes in self.term_index.items():
        if " " in term:
            # Multi-word term — compile regex once
            pattern = re.compile(
                r'\b' + re.escape(term) + r'\b',
                re.IGNORECASE
            )
            self.multiword_patterns.append((pattern, codes))
        else:
            self.singleword_terms[term] = codes

    log.info(
        "Term index: %d single-word terms, %d multi-word patterns",
        len(self.singleword_terms),
        len(self.multiword_patterns),
    )

    # ── Index 2: Category_Code → [IntentRecord] ───────────
    # Which intents are possible for a given category?
    self.category_intents: dict[str, list[IntentRecord]] = defaultdict(list)
    for record in self.intent_records:
        # Intent_Code can equal Category_Code (e.g. Billing)
        # or be a variant (e.g. Billing_BS, Billing_DM)
        # Match by prefix OR exact equality
        for cat_code in self.category_terms:
            if (record.intent_code == cat_code or
                record.intent_code.startswith(cat_code + "_") or
                cat_code.startswith(record.intent_code)):
                self.category_intents[cat_code].append(record)
                break
        else:
            # No category match — add under its own code
            self.category_intents[record.intent_code].append(record)

    # ── Index 3: email address → [IntentRecord] ───────────
    # Which intents route to a given email address?
    self.email_intents: dict[str, list[IntentRecord]] = defaultdict(list)
    for record in self.intent_records:
        self.email_intents[record.email].append(record)

    # ── Index 4: intent_code → IntentRecord ───────────────
    self.intent_by_code: dict[str, IntentRecord] = {
        r.intent_code: r for r in self.intent_records
    }

    # ── Index 5: domain → [IntentRecord] (for domain matching)
    self.domain_intents: dict[str, list[IntentRecord]] = defaultdict(list)
    for record in self.intent_records:
        domain = record.email.split("@")[-1] if "@" in record.email else record.email
        self.domain_intents[domain].append(record)

def _log_summary(self):
    log.info("─" * 50)
    log.info("Knowledge Base Summary:")
    log.info("  Categories       : %d", len(self.category_terms))
    log.info("  Total keywords   : %d", sum(len(t) for t in self.category_terms.values()))
    log.info("  Intent codes     : %d", len(self.intent_by_code))
    log.info("  Email addresses  : %d", len(self.email_intents))
    log.info("  Domains          : %d", len(self.domain_intents))
    log.info("─" * 50)
```

# ══════════════════════════════════════════════════════════════

# 2. STAGE 1 — KEYWORD MATCH

# ══════════════════════════════════════════════════════════════

def stage1_keyword_match(
email_text: str,
kb: IntentKnowledgeBase,
) -> tuple[list[str], dict[str, list[str]]]:
“””
Scan email body + subject for category keywords.

```
Returns:
  matched_categories : list of Category_Code(s) found
  keyword_hits       : {Category_Code: [matched_terms]} for audit/debug
"""
text_lower = email_text.lower()

# Tokenise for single-word lookup
tokens = set(re.findall(r'\b[a-z0-9]+\b', text_lower))

# Score each category by how many of its terms appear
category_scores: dict[str, int]       = defaultdict(int)
keyword_hits:    dict[str, list[str]] = defaultdict(list)

# ── Fast path: single-word terms via set intersection ─────
for token in tokens:
    if token in kb.singleword_terms:
        for cat_code in kb.singleword_terms[token]:
            category_scores[cat_code] += 1
            keyword_hits[cat_code].append(token)

# ── Slower path: multi-word terms via regex ────────────────
for pattern, codes in kb.multiword_patterns:
    matches = pattern.findall(text_lower)
    if matches:
        for code in codes:
            weight = 2   # multi-word match is stronger signal
            category_scores[code] += weight * len(matches)
            keyword_hits[code].extend(matches)

# ── Weight by term length if configured ───────────────────
if KEYWORD_SCORE_MODE == "weighted":
    for code in list(category_scores.keys()):
        # Longer terms = more specific = higher weight
        avg_len = (
            sum(len(t) for t in keyword_hits[code]) /
            max(len(keyword_hits[code]), 1)
        )
        category_scores[code] = int(category_scores[code] * (avg_len / 5))

# Return categories with at least 1 hit, sorted by score
matched = sorted(
    [c for c, s in category_scores.items() if s > 0],
    key=lambda c: category_scores[c],
    reverse=True,
)

log.debug(
    "Stage 1 — found %d categories: %s",
    len(matched),
    [(c, category_scores[c]) for c in matched[:5]]
)

return matched, dict(keyword_hits)
```

# ══════════════════════════════════════════════════════════════

# 3. STAGE 2 — INTENT LOOKUP

# ══════════════════════════════════════════════════════════════

def stage2_intent_lookup(
matched_categories: list[str],
kb: IntentKnowledgeBase,
) -> list[IntentRecord]:
“””
For each matched Category_Code, retrieve all possible IntentRecords.

```
Returns: flat list of candidate IntentRecord objects (may contain
         multiple intents per category, e.g. Billing, Billing_BS, Billing_DM)
"""
seen_codes = set()
candidates: list[IntentRecord] = []

for cat_code in matched_categories:
    records = kb.category_intents.get(cat_code, [])

    for record in records:
        if record.intent_code not in seen_codes:
            seen_codes.add(record.intent_code)
            candidates.append(record)

log.debug(
    "Stage 2 — %d candidate intents: %s",
    len(candidates),
    [r.intent_code for r in candidates]
)

return candidates
```

# ══════════════════════════════════════════════════════════════

# 4. STAGE 3 — EMAIL PRUNE

# ══════════════════════════════════════════════════════════════

def stage3_email_prune(
candidates: list[IntentRecord],
email_to:   str,
kb: IntentKnowledgeBase,
) -> tuple[list[IntentRecord], str]:
“””
Prune candidate intents by matching the email’s “To” address
against the email address stored with each IntentRecord.

```
Match modes:
  exact  — full email address must match  (businesssupport@hydroone.com)
  domain — just the @domain must match    (@hydroone.com)
  fuzzy  — substring match                (hydroone)

Returns:
  pruned    : list of IntentRecord that survived the prune
  match_src : string describing what matched
"""
if not email_to:
    return candidates, "keyword_only"   # no To address → skip prune

email_to_clean = email_to.strip().lower()

# ── Exact match ────────────────────────────────────────────
if EMAIL_MATCH_MODE == "exact":
    pruned = [r for r in candidates if r.email == email_to_clean]
    if pruned:
        return pruned, "keyword+email_exact"

# ── Domain match ───────────────────────────────────────────
elif EMAIL_MATCH_MODE == "domain":
    to_domain = email_to_clean.split("@")[-1] if "@" in email_to_clean else ""
    pruned = [
        r for r in candidates
        if r.email.split("@")[-1] == to_domain
    ]
    if pruned:
        return pruned, "keyword+email_domain"

# ── Fuzzy / substring match ─────────────────────────────────
elif EMAIL_MATCH_MODE == "fuzzy":
    pruned = [r for r in candidates if email_to_clean in r.email or r.email in email_to_clean]
    if pruned:
        return pruned, "keyword+email_fuzzy"

# ── No email match found — return all candidates unpruned ──
# (keyword match alone is still useful signal)
log.debug(
    "Stage 3 — no email match for '%s' among %d candidates, returning all",
    email_to_clean, len(candidates)
)
return candidates, "keyword_only"
```

# ══════════════════════════════════════════════════════════════

# 5. FULL PIPELINE

# ══════════════════════════════════════════════════════════════

def run_pipeline(
email_guid:    str,
email_subject: str,
email_body:    str,
email_to:      str,
kb:            IntentKnowledgeBase,
llm_fallback=  None,
) -> PipelineResult:
“””
Run all three stages for one email.

```
Flow:
  Stage 1: keyword match  →  Category_Code(s)
  Stage 2: intent lookup  →  candidate Intent_Code(s)
  Stage 3: email prune    →  final Intent_Code(s)
  Fallback: LLM           →  if pipeline returns nothing
"""
result = PipelineResult(guid=email_guid)

# Combine subject + body for keyword search
email_text = f"{email_subject or ''} {email_body or ''}"

# ── Stage 1 ────────────────────────────────────────────────
matched_cats, keyword_hits = stage1_keyword_match(email_text, kb)
result.matched_categories  = matched_cats
result.keyword_hits        = keyword_hits

if not matched_cats:
    log.debug("Stage 1: no keyword matches — going to fallback")
    result.final_intents = _handle_no_match(
        email_text, email_to, kb, llm_fallback, result
    )
    return result

# ── Stage 2 ────────────────────────────────────────────────
candidates = stage2_intent_lookup(matched_cats, kb)
result.candidate_intents = [r.intent_code for r in candidates]

if not candidates:
    log.debug("Stage 2: no candidate intents found")
    result.final_intents = ["Unclassified"]
    result.match_source  = "keyword_no_intent"
    return result

# ── Stage 3 ────────────────────────────────────────────────
pruned, match_source = stage3_email_prune(candidates, email_to, kb)
result.match_source  = match_source

if pruned:
    result.final_intents = _deduplicate([r.intent_code for r in pruned])
    result.confidence    = "high" if match_source != "keyword_only" else "medium"
else:
    # Email prune eliminated everything — fall back to candidates
    result.final_intents = _deduplicate([r.intent_code for r in candidates])
    result.match_source  = "keyword_only"
    result.confidence    = "medium"

return result
```

def _deduplicate(codes: list[str]) -> list[str]:
seen, out = set(), []
for c in codes:
if c not in seen:
seen.add(c)
out.append(c)
return out

def _handle_no_match(
email_text: str,
email_to:   str,
kb:         IntentKnowledgeBase,
llm_fallback,
result:     PipelineResult,
) -> list[str]:
“””
Called when Stage 1 finds zero keyword matches.
Tries:
1. Email-address-only lookup (maybe “To” alone identifies intent)
2. LLM fallback if configured
3. Unclassified
“””

```
# Try email-address-only lookup first
if email_to:
    email_clean = email_to.strip().lower()
    records = kb.email_intents.get(email_clean, [])
    if records:
        log.debug("No keywords but email address matched %d intents", len(records))
        result.match_source = "email_only"
        result.confidence   = "medium"
        return _deduplicate([r.intent_code for r in records])

# LLM fallback
if llm_fallback and USE_LLM_FALLBACK:
    log.info("No keyword/email match — calling LLM fallback")
    result.match_source = "llm_fallback"
    result.confidence   = "low"
    return llm_fallback(email_text)

result.match_source = "unmatched"
result.confidence   = "low"
return ["Unclassified"]
```

# ══════════════════════════════════════════════════════════════

# 6. OPTIONAL LLM FALLBACK

# ══════════════════════════════════════════════════════════════

def build_llm_fallback(kb: IntentKnowledgeBase):
“””
Minimal LLM fallback — only called when the deterministic
pipeline finds nothing. Prompt is compact because we only
send a short list of all valid codes (no terms needed —
the LLM already knows what billing means).
“””
try:
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field as PydanticField

```
    class FallbackResult(BaseModel):
        intent_codes: list[str] = PydanticField(description="Matched intent codes")

    llm = LlamaCpp(
        model_path   = GGUF_MODEL_PATH,
        n_ctx        = N_CTX,
        n_gpu_layers = N_GPU_LAYERS,
        max_tokens   = 100,
        temperature  = 0.05,
        verbose      = False,
        stop         = ["</s>", "[INST]"],
    )

    valid_codes = list(kb.intent_by_code.keys())
    codes_str   = " | ".join(valid_codes)

    parser  = JsonOutputParser(pydantic_object=FallbackResult)
    prompt  = PromptTemplate(
        input_variables=["email_text"],
        template=(
            "<s>[INST] Classify this email into one of these intent codes:\n"
            f"{codes_str}\n\n"
            "Return ONLY JSON: {{\"intent_codes\": [\"code1\"]}}\n\n"
            "Email: {{email_text}} [/INST]"
        ),
    )
    chain = prompt | llm | parser

    def fallback(email_text: str) -> list[str]:
        try:
            result = chain.invoke({"email_text": email_text[:800]})
            codes  = result.get("intent_codes", [])
            return [c for c in codes if c in kb.intent_by_code] or ["Unclassified"]
        except Exception as e:
            log.warning("LLM fallback failed: %s", e)
            return ["Unclassified"]

    log.info("LLM fallback ready.")
    return fallback

except ImportError:
    log.warning("LangChain/llama-cpp not installed — LLM fallback disabled.")
    return None
```

# ══════════════════════════════════════════════════════════════

# 7. SQL SERVER

# ══════════════════════════════════════════════════════════════

def get_engine():
driver = “ODBC Driver 17 for SQL Server”
cs = (
f”DRIVER={{{driver}}};SERVER={SQL_SERVER};”
f”DATABASE={SQL_DATABASE};Trusted_Connection=yes;”
if SQL_TRUSTED else
f”DRIVER={{{driver}}};SERVER={SQL_SERVER};”
f”DATABASE={SQL_DATABASE};UID=sa;PWD=password;”
)
return create_engine(f”mssql+pyodbc:///?odbc_connect={cs}”, fast_executemany=True)

def fetch_emails(engine, n: int) -> pd.DataFrame:
q = text(f”””
SELECT TOP (:n)
GUID,
[Subject],
[Body],
[To],
[From]
FROM {SQL_TABLE}
WHERE ([Predicted] IS NULL OR [Predicted] = ‘Unclassified’)
AND   ([Body] IS NOT NULL OR [Subject] IS NOT NULL)
ORDER BY [Date] DESC
“””)
with engine.connect() as conn:
df = pd.read_sql(q, conn, params={“n”: n})
log.info(“Fetched %d emails.”, len(df))
return df

def write_predictions(engine, results: list[PipelineResult]):
if DRY_RUN:
log.info(”[DRY RUN] Would write %d predictions:”, len(results))
for r in results:
log.info(
“  %-36s → %-40s [%s / %s]”,
str(r.guid)[:36],
“,”.join(r.final_intents),
r.match_source,
r.confidence,
)
return

```
q = text(f"""
    UPDATE {SQL_TABLE}
    SET [Predicted]     = :predicted,
        [PredictedDate] = :dt
    WHERE GUID = :guid
""")
now = datetime.now()
with engine.begin() as conn:
    for r in results:
        conn.execute(q, {
            "predicted": ",".join(r.final_intents),
            "dt":        now,
            "guid":      r.guid,
        })
log.info("Wrote %d predictions.", len(results))
```

# ══════════════════════════════════════════════════════════════

# 8. MAIN

# ══════════════════════════════════════════════════════════════

def main():
log.info(“═” * 60)
log.info(“Three-Stage Deterministic Intent Pipeline”)
log.info(“═” * 60)

```
# Build knowledge base
kb = IntentKnowledgeBase(CATEGORY_FILE, INTENT_FILE)

# Optional LLM fallback
llm_fallback = build_llm_fallback(kb) if USE_LLM_FALLBACK else None

# Fetch emails
engine    = get_engine()
emails_df = fetch_emails(engine, BATCH_SIZE)
if emails_df.empty:
    log.info("No emails to classify.")
    return

# Run pipeline
results: list[PipelineResult] = []
match_stats = defaultdict(int)

for _, row in tqdm(emails_df.iterrows(), total=len(emails_df), desc="Processing"):
    result = run_pipeline(
        email_guid    = row["GUID"],
        email_subject = str(row.get("Subject") or ""),
        email_body    = str(row.get("Body")    or ""),
        email_to      = str(row.get("To")      or ""),
        kb            = kb,
        llm_fallback  = llm_fallback,
    )
    results.append(result)
    match_stats[result.match_source] += 1

    log.info(
        "GUID %-36s → [%-30s]  src=%-25s  conf=%s",
        str(result.guid)[:36],
        ",".join(result.final_intents),
        result.match_source,
        result.confidence,
    )

# Write back
write_predictions(engine, results)

# Summary
log.info("═" * 60)
log.info("Done. %d emails processed.", len(results))
log.info("Match source breakdown:")
for src, count in sorted(match_stats.items(), key=lambda x: -x[1]):
    pct = 100 * count / max(len(results), 1)
    log.info("  %-30s %4d  (%5.1f%%)", src, count, pct)
multi = sum(1 for r in results if len(r.final_intents) > 1)
log.info("Multi-intent emails: %d / %d", multi, len(results))
log.info("═" * 60)
```

# ══════════════════════════════════════════════════════════════

# 9. TEST WITHOUT DATABASE

# ══════════════════════════════════════════════════════════════

def test_pipeline():
kb = IntentKnowledgeBase(CATEGORY_FILE, INTENT_FILE)

```
test_cases = [
    {
        "label":   "Billing — exact email match",
        "subject": "My bill is too high",
        "body":    "I received invoice bi84 case 3188. The bill breakdown looks wrong.",
        "to":      "businesssupport@hydroone.com",
    },
    {
        "label":   "Multi-intent — billing + account",
        "subject": "Bill query and address update",
        "body":    "My bill is incorrect. Also I need to update my account details and postal address.",
        "to":      "businesssupport@hydroone.com",
    },
    {
        "label":   "Undeliverable — auto reply",
        "subject": "Undeliverable: Re: Your query",
        "body":    "Mail delivery failed. This message could not be delivered.",
        "to":      "no-reply@hydroone.com",
    },
    {
        "label":   "Email-only match (no keywords)",
        "subject": "Hello",
        "body":    "Please help.",
        "to":      "omshelpdesk@hydroone.com",
    },
]

print("\n" + "═" * 65)
print("  THREE-STAGE PIPELINE TEST")
print("═" * 65)

for tc in test_cases:
    result = run_pipeline(
        email_guid    = "TEST-001",
        email_subject = tc["subject"],
        email_body    = tc["body"],
        email_to      = tc["to"],
        kb            = kb,
    )
    print(f"\n  Label   : {tc['label']}")
    print(f"  To      : {tc['to']}")
    print(f"  Subject : {tc['subject']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Stage 1 matched categories : {result.matched_categories}")
    print(f"  Stage 1 keyword hits       : {dict(list(result.keyword_hits.items())[:3])}")
    print(f"  Stage 2 candidate intents  : {result.candidate_intents}")
    print(f"  Stage 3 final intents      : {result.final_intents}")
    print(f"  Match source               : {result.match_source}")
    print(f"  Confidence                 : {result.confidence}")

print("\n" + "═" * 65 + "\n")
```

if **name** == “**main**”:
import sys
if len(sys.argv) > 1 and sys.argv[1] == “test”:
test_pipeline()
else:
main()
