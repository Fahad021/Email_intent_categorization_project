# Email Intent Classifier — User Guide

## Table of contents

1. [What this tool does](#1-what-this-tool-does)
2. [System requirements](#2-system-requirements)
3. [First-time setup](#3-first-time-setup)
4. [Files you need to provide](#4-files-you-need-to-provide)
5. [Configuring the classifier](#5-configuring-the-classifier)
6. [Running the classifier](#6-running-the-classifier)
7. [Understanding the output](#7-understanding-the-output)
8. [SQL Server mode](#8-sql-server-mode)
9. [Swapping the model, prompt, or knowledge base](#9-swapping-the-model-prompt-or-knowledge-base)
10. [Troubleshooting](#10-troubleshooting)
11. [Running the tests](#11-running-the-tests)
12. [Verifying your environment](#12-verifying-your-environment)

---

## 1. What this tool does

The Email Intent Classifier reads a batch of customer emails, sends each one to a
locally-running AI language model (no internet connection required), and outputs a
predicted intent category for every email — for example: *Billing*, *Outage*,
*New Connection*, *General Inquiry*.

All processing happens on your own machine.  No email content is sent to the cloud.

---

## 2. System requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 10/11 | Windows 10/11 |
| RAM | 8 GB | 16 GB |
| Disk space | 10 GB free | 20 GB free |
| Python | 3.10 | 3.11 (via Miniconda) |
| GPU | Not required | NVIDIA GPU (speeds up inference) |

> **No internet connection is required after the initial setup.**

---

## 3. First-time setup

### Option A — Automated (recommended)

Right-click `scripts\setup_env.ps1` and select **Run with PowerShell**.

Or open a terminal and run:

```
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

The script auto-detects whether you have **conda** (preferred) or plain **Python**,
creates the environment, installs all packages, creates required folders, and copies
the example config.  It then runs a verification check and tells you if anything is
missing.

**Using a Python environment you already have**

If you have a Python 3.10+ environment already set up (a venv, a conda env, or any
Python executable), you can skip environment creation entirely:

```powershell
# Use an existing venv (relative or absolute path)
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1 `
    -ExistingPython ".venv\Scripts\python.exe"

# Use an existing conda environment by name
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1 `
    -ExistingCondaEnv "my_nlp_env"
```

The script will validate the environment, then proceed from package installation
onward (skipping env creation).

**Installing llama-cpp-python from a local `.whl` file**

If your machine has no internet access, or a colleague has already downloaded the
wheel, you can supply it directly.  You can also drop any file matching
`llama_cpp_python*.whl` into the project root or a `wheels\` subfolder and the
script will find it automatically.

```powershell
# Explicit path
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1 `
    -LlamaWhl "C:\Downloads\llama_cpp_python-0.3.16-cp311-win_amd64.whl"

# Drop the .whl into wheels\ and just run normally — it is picked up automatically
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Parameters can be combined:

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1 `
    -ExistingPython ".venv\Scripts\python.exe" `
    -LlamaWhl "wheels\llama_cpp_python-0.3.16-cp311-win_amd64.whl"
```

### Option B — Manual (conda)

```bash
conda create -n email_intent python=3.11 -y
conda activate email_intent
conda install -c conda-forge llama-cpp-python=0.3.16 -y
pip install -r requirements.txt
```

### Option C — Manual (Python venv, no conda)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install llama-cpp-python==0.3.16
pip install -r requirements.txt
```

---

## 4. Files you need to provide

After setup, you must place two files into the project before the first run.

### 4.1 The AI model file

The classifier uses **Mistral 7B Instruct v0.2** in GGUF format (~4.4 GB).

1. Download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` from:
   https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
2. Save it to the `models\` folder inside this project.

> **Other GGUF models also work.**  Update `model_path` in `config.yaml` to point
> to a different file.

### 4.2 The knowledge base (Excel)

The knowledge base defines the list of intent categories the model can choose from.

| Sheet name | Required columns |
|---|---|
| `Merged_Knowledgebase` | `Reduced_Category`, `Merged_Terms` |

- **`Reduced_Category`** — The category name (e.g. `Billing`, `Outage`)
- **`Merged_Terms`** — Comma-separated keywords for that category  
  (e.g. `bill, invoice, overcharged, payment`)

Save it anywhere and set `kb_file` in `config.yaml` (see section 5).

### 4.3 The email data (parquet)

Your input emails must be in a **parquet file** with at least these three columns:

| Column | Description |
|---|---|
| `GUID` | Unique identifier per email |
| `Subject` | Email subject line |
| `Body` | Email body text |

Save it anywhere and set `parquet_in` in `config.yaml` (see section 5).

---

## 5. Configuring the classifier

All settings live in `config.yaml`.  The setup script creates this file
automatically by copying `config.example.yaml`.

Open `config.yaml` in any text editor and update at minimum these two lines:

```yaml
model_path: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
kb_file:    data/SPSS_Category_Concepts.xlsx
```

### Common settings explained

```yaml
# ── Where your data comes from ────────────────────────────
data_source: parquet          # "parquet" for file input  |  "sql" for SQL Server

parquet_in:  data/emails.parquet    # path to your input emails
parquet_out: ""                     # leave blank for auto-named output

# ── AI model settings ─────────────────────────────────────
n_ctx:       8192     # context window in tokens (increase for very long emails)
n_gpu_layers: 0       # set to 32+ to use a GPU for much faster processing
temperature: 0.05     # lower = more consistent (do not change for classification)

# ── Run behaviour ─────────────────────────────────────────
run_mode:  prod        # "prod" for normal use | "dev" for debugging
log_level: INFO        # INFO for normal use   | DEBUG for detailed logs
```

> **Tip:** The full list of settings with explanations is in `config.example.yaml`.

---

## 6. Running the classifier

### Activate your environment first

**conda:**
```bash
conda activate email_intent
```

**venv:**
```bash
.venv\Scripts\activate
```

### Run with config.yaml (simplest)

```bash
python main.py
```

This reads all settings from `config.yaml`.

### Run with CLI overrides

Any `config.yaml` setting can be overridden on the command line without editing the file:

```bash
# Use a different input file just for this run
python main.py --parquet-in data/new_batch.parquet

# Use a different config file (e.g. for experiments)
python main.py --config experiments/outage_focus.yaml

# Use GPU (32 layers) for faster processing
python main.py --n-gpu-layers 32

# Enable dry run (read and classify but do not write output)
python main.py --sql-dry-run

# Full debugging output
python main.py --mode dev --log-prompts
```

### Progress display

While running you will see a progress bar:

```
Classifying emails: 100%|████████████| 500/500 [04:12<00:00,  1.98 email/s]
```

Each email takes roughly 1–3 seconds on a modern CPU (much faster with a GPU).

---

## 7. Understanding the output

### Output parquet file

The classifier writes a new parquet file next to your input file:

```
data/emails_with_llm_predictions_20260407_120000.parquet
```

(Or the path you set in `parquet_out`.)

It contains all original columns **plus** the following new ones:

| Column | Description |
|---|---|
| `Predicted_Reduced_Category` | The predicted intent category |
| `Confidence` | How confident the model was: `high`, `medium`, `low` |
| `All_Intents` | All categories the model considered (JSON array) |
| `Parse_Status` | How the model's response was interpreted (see below) |
| `Processing_Status` | Full audit status for this row |
| `LLM_Called` | `True` if the model was run; `False` if the email was empty |
| `Error_Detail` | Error message if something went wrong (otherwise blank) |

### Parse status values

| Value | Meaning |
|---|---|
| `ok` | Model returned a valid, recognised category |
| `case_corrected` | Category matched after correcting capitalisation |
| `no_json` | Model's response contained no JSON — category defaulted to `Unclassified` |
| `bad_json` | Model's response contained malformed JSON |
| `empty_fields` | JSON was valid but category field was blank |
| `invalid_category` | Category returned was not in the knowledge base |
| `skipped_empty` | Email had no subject and no body — not sent to model |
| `error` | An exception occurred during inference |

### Manifest file

A summary file is written alongside the output:

```
data/emails_with_llm_predictions_20260407_120000.parquet.manifest.json
```

It records the run parameters, model details, row counts, and an integrity check
confirming that every input row has a corresponding output prediction.

### Telemetry records (optional)

If `no_records: false` (the default), a JSON file is written for every email
processed, stored in:

```
output/inference_records/000001__guid-abc__20260407T120000.json
```

Each file contains the full prompt, model response, token counts, latency, and
prediction for that email.  Useful for auditing and debugging.  Set `no_records: true`
in `config.yaml` to disable if disk space is a concern.

---

## 8. SQL Server mode

Instead of reading from a parquet file, the classifier can read directly from a
SQL Server database and write predictions back to it.

### Configuration

```yaml
data_source:  sql
sql_server:   YOUR_SERVER_NAME
sql_database: EMAIL
sql_table:    "[dbo].[Original_Email]"
sql_trusted:  true          # true = Windows Authentication
sql_batch:    100           # number of emails to fetch and process per run
sql_dry_run:  false         # true = classify but do NOT write back to database
```

### SQL credentials (non-Windows auth)

If `sql_trusted: false`, create a `.env` file in the project root:

```
SQL_UID=your_username
SQL_PWD=your_password
```

> **Never put passwords in `config.yaml`.**  The `.env` file is excluded from
> git automatically.

### What gets written back

The classifier updates these columns for each processed row:

- `Predicted` — the predicted intent category
- `PredictedDate` — timestamp of when the prediction was written

A parquet audit copy is also saved to `output/` as a backup.

---

## 9. Swapping the model, prompt, or knowledge base

Everything is designed to be swapped without changing any code.

### Swap the AI model

1. Download any GGUF-format instruction-tuned model.
2. Place it in `models\`.
3. Update `config.yaml`:
   ```yaml
   model_path: models/your-new-model.gguf
   ```

### Swap the system prompt

Edit `prompts\system_prompt.txt` directly, or point to a different file:

```yaml
prompt_file: prompts/billing_specialist_prompt.txt
```

The prompt file supports two placeholders which are filled in automatically at runtime:

- `{{ALLOWED_CATEGORIES}}` — replaced with the list of valid categories from the KB
- `{{KEYWORD_HINTS}}` — replaced with keyword hints per category

Set `prompt_file: ""` to use the built-in default prompt.

### Swap the knowledge base

Replace the Excel file with one that has new categories:

```yaml
kb_file: data/new_categories.xlsx
```

The sheet must be named `Merged_Knowledgebase` with columns `Reduced_Category`
and `Merged_Terms`.

### Run multiple experiments

Create separate config files and switch between them:

```bash
python main.py --config experiments/billing_only.yaml
python main.py --config experiments/full_categories.yaml
```

---

## 10. Troubleshooting

### "Model file not found"

Confirm the GGUF file is in `models\` and that `model_path` in `config.yaml` matches
the exact filename.  Run the verify script to check:

```bash
python scripts\verify_env.py --config config.yaml --check-model
```

### "Neither pyarrow nor fastparquet is installed"

```bash
conda activate email_intent
pip install pyarrow
```

### Many rows come back as `Unclassified`

1. Check that the category names in the KB match what the model would output.
   Category names with special characters or inconsistent capitalisation can cause
   mismatches.
2. Try `run_mode: dev` and `log_prompts: true` to inspect the prompts the model
   is receiving.
3. Lower `temperature` to `0.0` for more deterministic output.

### Processing is very slow

- Enable GPU offloading if you have an NVIDIA GPU:
  ```yaml
  n_gpu_layers: 32
  ```
- Reduce `max_tokens` if the model is generating long responses (256 is sufficient).
- Increase `n_batch` to `1024` if you have spare RAM.

### Log file is enormous

Set `log_prompts: false` and `no_records: true` in `config.yaml` to reduce output.

### "Access denied" when running the setup script

Use this command explicitly:

```
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

### Cannot import llama_cpp

The DLL must be loaded before pandas on Windows.  This is handled automatically
by the code.  If you are writing your own scripts, always put:

```python
from llama_cpp import Llama   # must come before: import pandas
import pandas as pd
```

---

## 11. Running the tests

The test suite requires no model file, no KB file, and no database connection.

```bash
conda activate email_intent

# Run all tests
python -m pytest tests/ -v

# Unit tests only (pure logic, fastest)
python -m pytest tests/unit -v

# Integration tests only (pipeline with mocked model)
python -m pytest tests/integration -v

# With coverage report
python -m pytest tests/ --cov=classifier --cov-report=term-missing
```

All 88 tests should pass in under 10 seconds.

---

## 12. Verifying your environment

Run this any time to confirm everything is in place:

```bash
python scripts\verify_env.py
```

To also check that the model file exists:

```bash
python scripts\verify_env.py --config config.yaml --check-model
```

A passing environment looks like:

```
  [PASS]  Python 3.11.15  (requires >= 3.10)
  [PASS]  llama-cpp-python==0.3.16
  [PASS]  pandas==3.0.2
  ...
  [PASS]  All checks passed — environment is ready
```

If any check fails, the script prints the exact command to fix it.
