"""
VariMAT QC Tool - Utility functions for file loading and variant key creation.
Supports .txt, .tsv, .txt.gz with efficient parsing for large files.
"""

from __future__ import annotations

import gc
import gzip
import io
import os
from typing import Any, Optional, Tuple

import pandas as pd
import polars as pl

# Required columns for variant key
REQUIRED_COLS = ["CHROM", "START", "REF", "ALT"]

# Sidecar cache filename suffix (same directory as source TSV)
_VARIMAT_PARQUET_SUFFIX = ".varimatqc.parquet"


def _varimat_parquet_sidecar_path(source_path: str) -> str:
    return os.path.abspath(source_path) + _VARIMAT_PARQUET_SUFFIX


def _polars_read_varimat_from_bytes(raw: bytes, nrows: Optional[int]) -> pl.DataFrame:
    # Polars fast path for TSV; infer_schema_length/ignore_errors match large messy genomics exports.
    buf = io.BytesIO(raw)
    kwargs: dict[str, Any] = {
        "separator": "\t",
        "comment_prefix": "#",
        "infer_schema_length": 10000,
        "ignore_errors": True,
    }
    if nrows is not None:
        kwargs["n_rows"] = nrows
    return pl.read_csv(buf, **kwargs)


def _polars_read_varimat_from_path(path: str, nrows: Optional[int]) -> pl.DataFrame:
    kwargs: dict[str, Any] = {
        "separator": "\t",
        "comment_prefix": "#",
        "infer_schema_length": 10000,
        "ignore_errors": True,
    }
    if nrows is not None:
        kwargs["n_rows"] = nrows
    # Polars reads compressed .gz directly on supported builds.
    return pl.read_csv(path, **kwargs)


def _strip_polars_column_names(pdf: pl.DataFrame) -> pl.DataFrame:
    # Match prior pandas behavior: strip whitespace from header names.
    rename = {c: str(c).strip() for c in pdf.columns}
    return pdf.rename(rename)


def _pl_to_pandas(pdf: pl.DataFrame) -> pd.DataFrame:
    return pdf.to_pandas()


def load_varimat(
    file_obj,
    filename: str,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a VariMAT file (.txt, .tsv, .txt.gz) into a DataFrame.
    Uses Polars for parsing, returns pandas for downstream compatibility.

    Parameters
    ----------
    file_obj : file-like or UploadedFile
        File object (e.g. Streamlit UploadedFile with .read() and .seek(0)).
    filename : str
        Original filename (used to detect .gz).
    nrows : int, optional
        Max rows to read (None = all).

    Returns
    -------
    pd.DataFrame
        Loaded VariMAT data with standardized column names (strip whitespace).
    """
    is_gz = filename.lower().endswith(".gz")
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    raw = file_obj.read()
    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        return pd.DataFrame()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    try:
        if is_gz:
            decompressed = gzip.decompress(raw)
            pl_df = _polars_read_varimat_from_bytes(decompressed, nrows)
        else:
            decompressed = None
            pl_df = _polars_read_varimat_from_bytes(raw, nrows)
        pl_df = _strip_polars_column_names(pl_df)
        df = _pl_to_pandas(pl_df)
        del pl_df
        if is_gz:
            del decompressed
        gc.collect()  # force free polars memory before returning
    except Exception as e:
        raise RuntimeError(f"Parse error: {e}") from e

    if df is None:
        return pd.DataFrame()
    return df


def create_variant_key(df: pd.DataFrame) -> pd.Series:
    """
    Create variant identifier: CHROM_START_REF_ALT.
    Used for unique variant-level comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns CHROM, START, REF, ALT.

    Returns
    -------
    pd.Series
        variant_key for each row.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns for variant_key: {missing}")
    chrom = df["CHROM"].astype(str).str.strip()
    start = df["START"].astype(str).str.strip()
    ref = df["REF"].astype(str).str.strip()
    alt = df["ALT"].astype(str).str.strip()
    return chrom + "_" + start + "_" + ref + "_" + alt


def safe_load_varimat(file_obj, filename: str, nrows: Optional[int] = None) -> tuple:
    """
    Load VariMAT file with error handling. Returns (df, error_message).
    If error_message is not None, df may be None.
    """
    try:
        df = load_varimat(file_obj, filename, nrows=nrows)
        if df.empty:
            return None, "File is empty."
        for col in REQUIRED_COLS:
            if col not in df.columns:
                return None, f"Missing required column: {col}. Found: {list(df.columns)[:20]}..."
        return df, None
    except Exception as e:
        return None, str(e)


def safe_load_varimat_from_path(path: str, nrows: Optional[int] = None) -> tuple:
    """
    Load a VariMAT file directly from a local path. Use when the app runs on
    the same machine as the files to avoid browser upload (much faster).

    Parquet sidecar: first full read writes ``<path>.varimatqc.parquet``; later loads
    use it if newer than the source mtime. Skipped when nrows is set (partial read).

    Returns (df, error_message). If error_message is not None, df is None.
    """
    try:
        path = os.path.expanduser(str(path).strip())
    except Exception:
        return None, "Invalid path"
    if not path or not os.path.isfile(path):
        return None, f"Not a file or not found: {path}"

    cache_path = _varimat_parquet_sidecar_path(path)
    try:
        src_mtime = os.path.getmtime(path)
    except OSError as e:
        return None, f"Cannot stat file: {e}"

    # Use sidecar only for full-file loads so cached row counts always match the source.
    if nrows is None:
        try:
            if os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= src_mtime:
                pl_df = pl.read_parquet(cache_path)
                pl_df = _strip_polars_column_names(pl_df)
                df = _pl_to_pandas(pl_df)
                if not df.empty and all(c in df.columns for c in REQUIRED_COLS):
                    del pl_df
                    gc.collect()  # force free polars memory before returning
                    return df, None
                del pl_df
                gc.collect()  # force free polars memory before continuing
        except Exception:
            pass

    try:
        pl_df = _polars_read_varimat_from_path(path, nrows)
        pl_df = _strip_polars_column_names(pl_df)
        df = _pl_to_pandas(pl_df)
    except Exception as e:
        return None, str(e)

    if df.empty:
        del pl_df
        gc.collect()
        return None, "File is empty."
    for col in REQUIRED_COLS:
        if col not in df.columns:
            del pl_df
            gc.collect()
            return None, f"Missing required column: {col}. Found: {list(df.columns)[:20]}..."

    if nrows is None:
        try:
            pl_df.write_parquet(cache_path, compression="snappy")
            # Verify write succeeded
            if os.path.isfile(cache_path):
                pass  # cache hit next time — no reparse needed
        except Exception:
            pass

    del pl_df
    gc.collect()
    return df, None


def load_varimat_path_worker(args: Tuple[int, str]) -> Tuple[int, str, str, Optional[pd.DataFrame], Optional[str], float]:
    """
    Thread/process pool entry point for path-based loads.
    Defined in this module so multiprocessing “spawn” can pickle the target if needed.
    """
    order, path = args
    path = os.path.expanduser(str(path).strip())
    if not path or not os.path.isfile(path):
        return (
            order,
            path,
            os.path.basename(path) or "unknown",
            None,
            f"File not found: {path}",
            0.0,
        )
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    df, err = safe_load_varimat_from_path(path)
    name = os.path.basename(path)
    return (order, path, name, df, err, size_mb)
