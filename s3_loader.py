"""
S3 helpers for VariMAT QC Dashboard: list, download to TMPDIR, parquet sidecar via utils.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import streamlit as st

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment,misc]
    BotoCoreError = ClientError = Exception  # type: ignore[misc,assignment]

# Same suffix as utils.safe_load_varimat_from_path sidecar (do not change).
_VARIMATQC_PARQUET_SUFFIX = ".varimatqc.parquet"


def _get_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve from Streamlit secrets first, then environment."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            v = st.secrets[key]
            if v is not None and str(v).strip() != "":
                return str(v).strip()
    except Exception:
        pass
    v = os.environ.get(key)
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    return default


def get_s3_config() -> Optional[dict[str, str]]:
    """Return bucket/prefix when S3_BUCKET is set; otherwise None (hide S3 UI)."""
    bucket = _get_str("S3_BUCKET")
    if not bucket:
        return None
    prefix = _get_str("S3_PREFIX", "") or ""
    return {"bucket": bucket, "prefix": prefix}


def get_s3_client():
    """Build boto3 S3 client from env / st.secrets."""
    if boto3 is None:
        raise RuntimeError(
            "boto3 is not installed. Install it with: pip install boto3>=1.34.0"
        )
    access = _get_str("AWS_ACCESS_KEY_ID")
    secret = _get_str("AWS_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise RuntimeError(
            "AWS credentials missing: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "(environment variables or .streamlit/secrets.toml)."
        )
    region = _get_str("AWS_REGION", "us-east-1") or "us-east-1"
    endpoint = _get_str("AWS_ENDPOINT_URL")
    kwargs: dict[str, Any] = {
        "service_name": "s3",
        "region_name": region,
        "aws_access_key_id": access,
        "aws_secret_access_key": secret,
    }
    token = _get_str("AWS_SESSION_TOKEN")
    if token:
        kwargs["aws_session_token"] = token
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client(**kwargs)


def list_s3_files(bucket: str, prefix: str = "") -> list[dict[str, Any]]:
    """
    List objects under prefix; only .txt / .tsv / .gz; newest first.
    On error: return [] and set st.session_state['s3_list_error'].
    """
    st.session_state.pop("s3_list_error", None)
    try:
        client = get_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix or "")
        rows: list[dict[str, Any]] = []
        for page in pages:
            for obj in page.get("Contents") or []:
                key = obj.get("Key") or ""
                if not key or key.endswith("/"):
                    continue
                lk = key.lower()
                if not (lk.endswith(".txt") or lk.endswith(".tsv") or lk.endswith(".gz")):
                    continue
                size = float(obj.get("Size") or 0) / (1024 * 1024)
                lm = obj.get("LastModified")
                if lm is not None and hasattr(lm, "astimezone"):
                    lm_str = lm.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                else:
                    lm_str = str(lm) if lm is not None else ""
                rows.append({"key": key, "size_mb": round(size, 3), "last_modified": lm_str, "_lm": lm})
        _epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        rows.sort(key=lambda r: r["_lm"] if r["_lm"] is not None else _epoch, reverse=True)
        for r in rows:
            r.pop("_lm", None)
        return rows
    except Exception as e:
        st.session_state["s3_list_error"] = str(e)
        return []


class _S3Progress:
    """boto3 transfer Callback: byte increments -> Streamlit progress bar."""

    def __init__(self, total: int, bar: Any):
        self._seen = 0
        self._total = total
        self._bar = bar

    def __call__(self, bytes_amount: int) -> None:
        self._seen += bytes_amount
        if self._total:
            self._bar.progress(min(self._seen / self._total, 1.0))


def _tmp_dir() -> str:
    return os.environ.get("TMPDIR") or "/tmp"


def download_s3_file(
    bucket: str,
    key: str,
    file_progress: Optional[Any] = None,
) -> str:
    """
    Download object to TMPDIR/basename if missing or local file older than 1 hour.
    Refresh parquet sidecar via utils.safe_load_varimat_from_path after a fresh download.
    """
    if boto3 is None:
        raise RuntimeError(
            "boto3 is not installed. Install it with: pip install boto3>=1.34.0"
        )
    fname = os.path.basename(key) or "download.varimat"
    local_path = os.path.join(_tmp_dir(), fname)
    cache_path = os.path.abspath(local_path) + _VARIMATQC_PARQUET_SUFFIX

    now = time.time()
    need_download = True
    if os.path.isfile(local_path):
        try:
            age = now - os.path.getmtime(local_path)
            if age < 3600:
                need_download = False
        except OSError:
            need_download = True

    bar = file_progress if file_progress is not None else st.progress(0)
    try:
        client = get_s3_client()
        if need_download:
            head = client.head_object(Bucket=bucket, Key=key)
            total = int(head.get("ContentLength") or 0)
            progress_cb: Optional[Callable[[int], None]] = _S3Progress(total, bar) if total > 0 else None
            client.download_file(bucket, key, local_path, Callback=progress_cb)
            bar.progress(1.0)
            # PERF: same parquet sidecar contract as utils.safe_load_varimat_from_path
            from utils import safe_load_varimat_from_path

            _df, err = safe_load_varimat_from_path(local_path)
            if err:
                raise RuntimeError(f"Downloaded file is not a valid VariMAT: {err}")
        else:
            bar.progress(1.0)
            # Ensure sidecar exists for repeat sessions (no full re-download).
            if not os.path.isfile(cache_path):
                from utils import safe_load_varimat_from_path

                _df, err = safe_load_varimat_from_path(local_path)
                if err:
                    raise RuntimeError(f"Cached file invalid: {err}")
    except RuntimeError:
        raise
    except (ClientError, BotoCoreError, OSError) as e:
        raise RuntimeError(f"S3 download failed for “{key}”: {e}") from e
    except Exception as e:
        raise RuntimeError(f"S3 download failed for “{key}”: {e}") from e

    return local_path
