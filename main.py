"""
main.py — Canonical entry point for the Email Intent Classifier.

Usage:
    python main.py                              # use config.yaml
    python main.py --config experiments/x.yaml  # swap config file
    python main.py --temperature 0.01           # override a single value
    python main.py --mode dev --log-prompts     # dev run with prompt logging

Any CLI flag overrides the corresponding value in the config file.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml is required. Run: pip install pyyaml")

from dotenv import load_dotenv

from claude import Config, run, ensure_dir

load_dotenv()


# --------------------------------------------------
# Config loader
# --------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


# --------------------------------------------------
# CLI
# --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Email Intent Classifier — YAML-driven entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file.",
    )

    # Every override below is OPTIONAL — omit to use the value from config.yaml.
    # The dest names match the Config dataclass field names exactly.

    paths = p.add_argument_group("Path overrides")
    paths.add_argument("--model",       dest="model_path",  default=None)
    paths.add_argument("--kb",          dest="kb_file",     default=None)
    paths.add_argument("--prompt-file", dest="prompt_file", default=None)

    data = p.add_argument_group("Data source overrides")
    data.add_argument("--data-source",  dest="data_source",  choices=["parquet", "sql"], default=None)
    data.add_argument("--in",           dest="parquet_in",   default=None)
    data.add_argument("--out",          dest="parquet_out",  default=None)
    data.add_argument("--out-col",      dest="out_col",      default=None)

    sql = p.add_argument_group("SQL overrides")
    sql.add_argument("--sql-server",   dest="sql_server",   default=None)
    sql.add_argument("--sql-database", dest="sql_database", default=None)
    sql.add_argument("--sql-table",    dest="sql_table",    default=None)
    sql.add_argument("--sql-trusted",  dest="sql_trusted",  action="store_true", default=None)
    sql.add_argument("--sql-batch",    dest="sql_batch",    type=int, default=None)
    sql.add_argument("--sql-dry-run",  dest="sql_dry_run",  action="store_true", default=None)

    llm = p.add_argument_group("LLM overrides")
    llm.add_argument("--n-ctx",          dest="n_ctx",          type=int,   default=None)
    llm.add_argument("--n-gpu-layers",   dest="n_gpu_layers",   type=int,   default=None)
    llm.add_argument("--n-threads",      dest="n_threads",      type=int,   default=None)
    llm.add_argument("--n-batch",        dest="n_batch",        type=int,   default=None)
    llm.add_argument("--max-tokens",     dest="max_tokens",     type=int,   default=None)
    llm.add_argument("--temperature",    dest="temperature",    type=float, default=None)
    llm.add_argument("--top-p",          dest="top_p",          type=float, default=None)
    llm.add_argument("--top-k",          dest="top_k",          type=int,   default=None)
    llm.add_argument("--repeat-penalty", dest="repeat_penalty", type=float, default=None)
    llm.add_argument("--max-keywords",   dest="max_keywords",   type=int,   default=None)

    gguf = p.add_argument_group("GGUF overrides")
    gguf.add_argument("--use-mlock",  dest="use_mlock",  action="store_true", default=None)
    gguf.add_argument("--flash-attn", dest="flash_attn", action="store_true", default=None)

    inf = p.add_argument_group("Inference overrides")
    inf.add_argument("--timeout", dest="infer_timeout_sec", type=int, default=None)
    inf.add_argument("--retries", dest="infer_retries",     type=int, default=None)

    lg = p.add_argument_group("Logging overrides")
    lg.add_argument("--log-file",    dest="log_file",    default=None)
    lg.add_argument("--log-level",   dest="log_level",   choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    lg.add_argument("--log-json",    dest="json_logs",   action="store_true", default=None)
    lg.add_argument("--no-redact",   dest="no_redact",   action="store_true", default=None)
    lg.add_argument("--log-prompts", dest="log_prompts", action="store_true", default=None)
    lg.add_argument("--no-records",  dest="no_records",  action="store_true", default=None)
    lg.add_argument("--record-dir",  dest="record_dir",  default=None)

    p.add_argument("--mode", dest="run_mode", choices=["dev", "test", "prod"], default=None)

    return p


# --------------------------------------------------
# Merge: YAML base → CLI overrides
# --------------------------------------------------

def _apply_overrides(cfg_dict: dict, args: argparse.Namespace) -> dict:
    """Replace cfg_dict values with any non-None CLI args."""
    bool_flags = {"use_mlock", "flash_attn", "json_logs", "log_prompts", "no_records",
                  "sql_trusted", "sql_dry_run"}
    no_redact = getattr(args, "no_redact", None)

    for field, value in vars(args).items():
        if field == "config":
            continue
        if field == "no_redact":
            if value:
                cfg_dict["redact_logs"] = False
            continue
        if field in bool_flags:
            if value:  # action="store_true" — only override when the flag was passed
                cfg_dict[field] = True
        elif value is not None:
            cfg_dict[field] = value

    return cfg_dict


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: config file not found: {config_path}")

    cfg_dict = load_yaml_config(config_path)
    cfg_dict = _apply_overrides(cfg_dict, args)

    # Auto-generate log file path if not set
    if not cfg_dict.get("log_file"):
        logs_dir = "logs"
        ensure_dir(logs_dir)
        cfg_dict["log_file"] = os.path.join(
            logs_dir,
            f"email_intent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

    # Build Config — Config.__post_init__ validates required fields
    try:
        cfg = Config(**{k: v for k, v in cfg_dict.items() if not k.startswith("#")})
    except TypeError as e:
        sys.exit(f"ERROR: unknown config key — {e}")

    run(cfg)


if __name__ == "__main__":
    main()
