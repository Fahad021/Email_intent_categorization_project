from __future__ import annotations

import logging
import os

import pandas as pd

from .config import Config


# --------------------------------------------------
# PARQUET HELPERS
# --------------------------------------------------

def read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
    except Exception:
        df.to_parquet(path, engine="fastparquet", index=False)


# --------------------------------------------------
# SQL HELPERS
# --------------------------------------------------

def _build_sql_engine(cfg: Config):
    """Build a SQLAlchemy engine from cfg SQL fields (+ env vars for credentials)."""
    import urllib.parse
    from sqlalchemy import create_engine

    if cfg.sql_trusted:
        params = urllib.parse.quote_plus(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={cfg.sql_server};DATABASE={cfg.sql_database};"
            "Trusted_Connection=yes;"
        )
    else:
        uid = os.environ.get("SQL_UID", "")
        pwd = os.environ.get("SQL_PWD", "")
        params = urllib.parse.quote_plus(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={cfg.sql_server};DATABASE={cfg.sql_database};"
            f"UID={uid};PWD={pwd};"
        )
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        fast_executemany=True,
    )


def read_from_sql(cfg: Config, log: logging.Logger) -> pd.DataFrame:
    """Fetch unclassified emails from SQL Server (Predicted IS NULL/empty, up to sql_batch rows)."""
    from sqlalchemy import text

    engine = _build_sql_engine(cfg)
    # TOP N cannot be parameterised in T-SQL; sql_batch is a validated int from Config
    query = text(
        f"SELECT TOP {int(cfg.sql_batch)} GUID, Subject, Body "
        f"FROM {cfg.sql_table} "
        f"WHERE Predicted IS NULL OR Predicted = '' "
        f"ORDER BY Date ASC"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info("sql_input_loaded", extra={"rows_in": len(df), "sql_table": cfg.sql_table})
    return df


def write_to_sql(df: pd.DataFrame, cfg: Config, log: logging.Logger) -> None:
    """Write predictions back to SQL Server row-by-row. Skipped when sql_dry_run=True."""
    if cfg.sql_dry_run:
        log.info("sql_dry_run_skip", extra={"rows": len(df), "sql_table": cfg.sql_table})
        return
    from sqlalchemy import text

    engine = _build_sql_engine(cfg)
    out_col = cfg.out_col
    stmt = text(
        f"UPDATE {cfg.sql_table} "
        f"SET Predicted = :pred, PredictedDate = GETDATE() "
        f"WHERE GUID = :guid"
    )
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(stmt, {"pred": row[out_col], "guid": row["GUID"]})
    log.info("sql_output_written", extra={"rows": len(df), "sql_table": cfg.sql_table})
