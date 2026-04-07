#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment verification script for the Hydro One Email Intent Classifier.

Run this after cloning the repo to confirm every dependency and path is in place:

    python scripts/verify_env.py
    python scripts/verify_env.py --config config.yaml
    python scripts/verify_env.py --config config.yaml --check-model

Exit codes:
    0 — all checks passed (or only warnings)
    1 — one or more required checks failed
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

# ── colour helpers ────────────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty() and os.name != "nt" or (
    os.name == "nt" and os.environ.get("TERM") not in (None, "")
)
_GREEN  = "\033[32m" if _USE_COLOUR else ""
_YELLOW = "\033[33m" if _USE_COLOUR else ""
_RED    = "\033[31m" if _USE_COLOUR else ""
_BOLD   = "\033[1m"  if _USE_COLOUR else ""
_RESET  = "\033[0m"  if _USE_COLOUR else ""

PASS  = f"{_GREEN}[PASS]{_RESET}"
WARN  = f"{_YELLOW}[WARN]{_RESET}"
FAIL  = f"{_RED}[FAIL]{_RESET}"
INFO  = f"[INFO]"
HEAD  = f"{_BOLD}{{title}}{_RESET}"


def _head(title: str) -> None:
    print(f"\n{_BOLD}{title}{_RESET}")
    print("-" * (len(title) + 2))


_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def warn(msg: str) -> None:
    print(f"  {WARN}  {msg}")
    _warnings.append(msg)


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")
    _failures.append(msg)


def info(msg: str) -> None:
    print(f"  {INFO}  {msg}")


# ── 1. Python Version ─────────────────────────────────────────────────────────

def check_python_version(min_major: int = 3, min_minor: int = 10) -> None:
    _head("Python version")
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (min_major, min_minor):
        ok(f"Python {ver_str}  (requires >= {min_major}.{min_minor})")
    else:
        fail(f"Python {ver_str}  (requires >= {min_major}.{min_minor})")
    info(f"Executable: {sys.executable}")


# ── 2. Required packages ──────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    # (import_name,       pip_name,               required)
    ("llama_cpp",         "llama-cpp-python",      True),
    ("pandas",            "pandas",                True),
    ("openpyxl",          "openpyxl",              True),
    ("tqdm",              "tqdm",                  True),
    ("yaml",              "pyyaml",                True),
    ("dotenv",            "python-dotenv",         True),
    ("pyarrow",           "pyarrow",               False),   # preferred parquet engine
    ("fastparquet",       "fastparquet",           False),   # fallback parquet engine
    ("pyodbc",            "pyodbc",                False),   # SQL Server optional
    ("sqlalchemy",        "sqlalchemy",            False),   # SQL Server optional
    ("psutil",            "psutil",                False),   # CPU thread detection
]


def check_packages() -> None:
    _head("Python packages")
    parquet_ok = False
    for import_name, pip_name, required in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            ok(f"{pip_name}=={version}")
            if import_name in ("pyarrow", "fastparquet"):
                parquet_ok = True
        except ImportError:
            if required:
                fail(f"{pip_name}  — not installed  (pip install {pip_name})")
            else:
                warn(f"{pip_name}  — not installed (optional)")

    if not parquet_ok:
        fail("Neither pyarrow nor fastparquet is installed — parquet I/O will not work")


# ── 3. classifier package integrity ───────────────────────────────────────────

CLASSIFIER_MODULES = [
    "classifier",
    "classifier.config",
    "classifier.utils",
    "classifier.logger",
    "classifier.kb",
    "classifier.prompt_builder",
    "classifier.model_loader",
    "classifier.inference",
    "classifier.data_io",
    "classifier.telemetry",
    "classifier.pipeline",
]


def check_classifier_package() -> None:
    _head("classifier package")
    # Make sure project root is on sys.path so relative imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for mod_name in CLASSIFIER_MODULES:
        try:
            importlib.import_module(mod_name)
            ok(mod_name)
        except ImportError as e:
            fail(f"{mod_name}  — import failed: {e}")
        except Exception as e:
            # Heavy modules (llama_cpp) may fail at runtime but import OK for structure
            warn(f"{mod_name}  — imported with warning: {e}")


# ── 4. Required folders ───────────────────────────────────────────────────────

REQUIRED_DIRS = ["data", "models", "output", "logs", "prompts"]


def check_folders(project_root: str) -> None:
    _head("Project folders")
    for d in REQUIRED_DIRS:
        path = os.path.join(project_root, d)
        if os.path.isdir(path):
            ok(f"{d}/")
        else:
            warn(f"{d}/  — missing, will be created on first run  (run: mkdir {d})")


# ── 5. Key project files ──────────────────────────────────────────────────────

REQUIRED_FILES = [
    ("requirements.txt",     True),
    ("config.example.yaml",  True),
    ("prompts/system_prompt.txt", False),
]


def check_project_files(project_root: str) -> None:
    _head("Project files")
    for rel_path, required in REQUIRED_FILES:
        path = os.path.join(project_root, rel_path)
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            ok(f"{rel_path}  ({size_kb:.1f} KB)")
        elif required:
            fail(f"{rel_path}  — missing")
        else:
            warn(f"{rel_path}  — missing (optional but recommended)")

    # config.yaml — warn if absent so user knows to copy from example
    config_path = os.path.join(project_root, "config.yaml")
    if os.path.isfile(config_path):
        ok("config.yaml")
    else:
        warn("config.yaml  — not found; copy config.example.yaml  (cp config.example.yaml config.yaml)")


# ── 6. YAML config validation ─────────────────────────────────────────────────

def check_config(config_path: str) -> dict:
    """Load and validate config.yaml. Returns the parsed dict (may be empty on failure)."""
    _head(f"Config file: {os.path.basename(config_path)}")
    if not os.path.isfile(config_path):
        fail(f"{config_path}  — file not found")
        return {}

    try:
        import yaml  # type: ignore
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as e:
        fail(f"Could not parse {config_path}: {e}")
        return {}

    required_keys = ["model_path", "kb_file"]
    for key in required_keys:
        if cfg.get(key):
            ok(f"{key} = {cfg[key]}")
        else:
            warn(f"{key}  — not set in config (required to run)")

    optional_keys = ["data_source", "prompt_file", "n_ctx", "n_threads"]
    for key in optional_keys:
        if key in cfg:
            ok(f"{key} = {cfg[key]}")

    return cfg


# ── 7. Model file ─────────────────────────────────────────────────────────────

def check_model_file(model_path: str) -> None:
    _head("GGUF model file")
    if not model_path:
        warn("model_path not configured — skipping model check")
        return
    if os.path.isfile(model_path):
        size_gb = os.path.getsize(model_path) / (1024 ** 3)
        ok(f"{model_path}  ({size_gb:.2f} GB)")
    else:
        fail(
            f"{model_path}  — file not found\n"
            "         Download the model and place it in the models/ folder.\n"
            "         See README.md for the download link."
        )


# ── 8. ODBC driver (optional — only relevant for SQL mode) ───────────────────

def check_odbc() -> None:
    _head("ODBC driver (SQL Server — optional)")
    try:
        import pyodbc  # type: ignore
        drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
        if drivers:
            ok(f"Found: {', '.join(drivers)}")
        else:
            warn(
                "pyodbc installed but no 'SQL Server' ODBC driver found.\n"
                "         Install 'ODBC Driver 17 for SQL Server' from Microsoft\n"
                "         if you plan to use data_source=sql."
            )
    except ImportError:
        info("pyodbc not installed — SQL mode unavailable (parquet mode still works)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the Hydro One Email Intent Classifier environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to config.yaml to validate (optional).",
    )
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Also check that the GGUF model file exists (requires --config or model_path in config.yaml).",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"\n{_BOLD}Hydro One Email Intent Classifier — Environment Check{_RESET}")
    print(f"Project root: {project_root}")

    check_python_version()
    check_packages()
    check_classifier_package()
    check_folders(project_root)
    check_project_files(project_root)

    cfg: dict = {}
    config_path = args.config or os.path.join(project_root, "config.yaml")
    if os.path.isfile(config_path):
        cfg = check_config(config_path)
    else:
        info(f"No config.yaml found at {config_path} — skipping config validation")

    if args.check_model:
        model_path = cfg.get("model_path", "")
        check_model_file(model_path)

    check_odbc()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'='*54}{_RESET}")
    if _failures:
        print(f"  {FAIL}  {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"          - {f}")
        if _warnings:
            print(f"  {WARN}  {len(_warnings)} warning(s) — see above for details")
        print(f"{_BOLD}{'='*54}{_RESET}\n")
        sys.exit(1)
    elif _warnings:
        print(f"  {PASS}  All required checks passed")
        print(f"  {WARN}  {len(_warnings)} warning(s) — see above for details")
        print(f"{_BOLD}{'='*54}{_RESET}\n")
        sys.exit(0)
    else:
        print(f"  {PASS}  All checks passed — environment is ready")
        print(f"{_BOLD}{'='*54}{_RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
