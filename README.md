# Email Intent Categorization

LLM-based email intent classifier for Hydro One. Classifies customer emails into intent categories using a local Mistral 7B GGUF model.

## Quick start

```bash
# 1. Activate the conda environment
conda activate email_intent

# 2. Copy and edit config
cp config.example.yaml config.yaml   # fill in model_path, kb_file, parquet_in

# 3. Run
python main.py
```

## Repo structure

```
.
├── main.py                  # Entry point — loads config.yaml, runs classifier
├── claude.py                # Core pipeline (model, inference, I/O, telemetry)
├── config.yaml              # Your active config (edit this)
├── config.example.yaml      # Fully documented reference copy
├── requirements.txt         # Python dependencies
│
├── prompts/
│   └── system_prompt.txt    # Swappable system prompt template
│
├── data/                    # Input parquet files (gitignored)
├── models/                  # GGUF model files (gitignored)
├── output/                  # Prediction parquet files (gitignored)
├── logs/                    # Run logs (gitignored)
│
└── archive/                 # Earlier prototype scripts (reference only)
```

## Configuration

All settings live in `config.yaml`. CLI flags override any YAML value.

```yaml
# Swap data source
data_source: parquet   # or: sql

# Swap prompt
prompt_file: prompts/system_prompt.txt   # or: "" for built-in

# Swap model
model_path: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### Key CLI overrides

```bash
# Use a different config file (experiments)
python main.py --config experiments/billing_focus.yaml

# Override temperature only
python main.py --temperature 0.0

# SQL mode with dry run
python main.py --data-source sql --sql-server MYSERVER --sql-dry-run

# Dev mode with full prompt logging
python main.py --mode dev --log-prompts
```

## Data source modes

| Mode | `data_source` | Input | Output |
|------|--------------|-------|--------|
| Parquet | `parquet` | `parquet_in` file | `parquet_out` file |
| SQL Server | `sql` | SQL table (unclassified rows) | Parquet audit copy + SQL `UPDATE` |

SQL credentials go in `.env` (never in `config.yaml`):

```
SQL_UID=myuser
SQL_PWD=mypassword
```

## Swapping parts

| Part | How to swap |
|------|-------------|
| Model | Set `model_path` in config.yaml |
| Prompt | Edit `prompts/system_prompt.txt` or point `prompt_file` at a new file |
| KB / categories | Set `kb_file` to a different Excel file |
| Data source | Set `data_source: sql` or `data_source: parquet` |
| Config file | `python main.py --config experiments/x.yaml` |

### Prompt template placeholders

`prompts/system_prompt.txt` uses two runtime-filled placeholders:

- `{{ALLOWED_CATEGORIES}}` — one `- Category Name` line per KB category
- `{{KEYWORD_HINTS}}` — one `- Category: kw1, kw2, ...` line per category

## Model

Tested with **Mistral 7B Instruct v0.2 Q4_K_M** (~4.4 GB GGUF).

Download: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

Place the file in `models/` and update `model_path` in `config.yaml`.

## Installation

```bash
conda create -n email_intent python=3.11
conda activate email_intent
conda install -c conda-forge llama-cpp-python  # pre-built binary
pip install -r requirements.txt
```

## Output columns

The classifier appends these columns to the output parquet:

| Column | Description |
|--------|-------------|
| `Predicted_Reduced_Category` | Predicted intent label |
| `Parse_Status` | JSON parse outcome (`ok`, `no_json`, etc.) |
| `Processing_Status` | Full audit status per row |
| `LLM_Called` | Whether LLM was invoked (False for empty emails) |
| `Confidence` | `high` / `medium` / `low` |
| `All_Intents` | JSON array of all plausible categories detected |
| `Error_Detail` | Exception message if inference failed |

A `.manifest.json` and per-row telemetry JSON files are written alongside the output.
