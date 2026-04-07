# Email Intent Categorization Project

Automatically classify incoming emails by intent using a local LLM (Mistral GGUF via `llama-cpp-python`). Results are written back to a SQL Server database.

## Overview

The project reads unclassified emails from a SQL Server table, builds a prompt using a known category/intent knowledge-base (Excel files), and predicts the intent via a locally-run quantised Mistral model. Three implementations are provided:

| File | Description |
|---|---|
| `Script.py` | Primary implementation — env-based config, SQL Server integration, batch processing |
| `claude.py` | Full telemetry version (v4.7.0 of the file) — CLI arguments, billing-priority fixes, Mistral best-practices prompt |
| `Non_nlp_soln.py` | Rule/keyword-based fallback — no LLM required |
| `Keyphrase_finder.py` | Utility to extract key phrases from email bodies |

## Requirements

- Python 3.9+
- A Mistral GGUF model file (e.g. [`mistral-7b-instruct-v0.2.Q4_K_M.gguf`](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF))
- SQL Server accessible via ODBC (Windows Auth or SQL Auth)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description | Default |
|---|---|---|
| `GGUF_MODEL_PATH` | Path to the `.gguf` model file | *(required)* |
| `CATEGORY_FILE` | Path to the category Excel file | *(required)* |
| `INTENT_FILE` | Path to the intent Excel file | *(required)* |
| `SQL_SERVER` | SQL Server hostname | *(required)* |
| `SQL_DATABASE` | Database name | `EMAIL` |
| `SQL_TABLE` | Table name | `[dbo].[Original_Email]` |
| `SQL_TRUSTED` | Use Windows Auth | `true` |
| `BATCH_SIZE` | Emails per run | `10` |
| `DRY_RUN` | Predict without writing to DB | `false` |
| `N_GPU_LAYERS` | GPU layers for llama.cpp | `0` |
| `N_CTX` | Context window (tokens) | `4096` |
| `MAX_TOKENS` | Max tokens in LLM response | `100` |
| `TEMPERATURE` | Sampling temperature | `0.05` |

## Running

```bash
# Using Script.py (env/.env config)
python Script.py

# Using claude.py (CLI arguments)
python claude.py --model /path/to/model.gguf \
                 --categories /path/to/categories.xlsx \
                 --intents /path/to/intents.xlsx \
                 --server your-sql-server
```

## Branch Strategy

This repository uses the following branching model:

```
main / master
  └── feature/billing-priority-fix   ← isolated fix: billing keywords were not
  │                                     triggering as mandatory signals in system
  │                                     prompts (see commit d2716f8)
  └── feature/<topic>                ← other feature branches
```

### Moving the last commit to a different branch

If you have committed to the wrong branch and want to move only the latest commit to a new branch while continuing work on the current one:

```bash
# 1. Create a new branch pointing at your current HEAD (preserves the commit)
git branch feature/my-fix

# 2. Remove the commit from the current branch (soft keeps your changes staged)
git reset --soft HEAD~1

# 3. Continue working on the current branch — the commit now lives only on feature/my-fix
```

> **Tip:** Use `git reset --hard HEAD~1` instead of `--soft` if you want to discard the staged changes entirely. Use `--soft` to keep them ready for a new commit.
