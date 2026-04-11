"""
Cloud vs On-Prem executive validation: on-target, optional gene filter, transcript-level cell concordance.
Vectorized comparisons for large VariMAT files.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# Required for row identity
KEY_COLS = ["CHROM", "START", "REF", "ALT", "ENS_TRANS_ID"]
VARIANT_LOCATION_COL = "VARIANT_LOCATION"
ONTARGET_VALUE = "ONTARGET"

# Cap rows stored in mismatch_df (counts remain exact); keeps UI/CSV/PDF memory bounded
MAX_MISMATCH_ROWS = 250_000

# CONCORDANCE — columns always excluded from comparison (case-insensitive)
EXCLUDE_COLS = {"EI_TOTAL", "EI_total", "ei_total"}

# CONCORDANCE — VarTk score uses tighter tolerance than the default 10%
VARTK_COLS = {"VARTK_SCORE", "VarTk_score", "VARTK", "vartk_score"}
VARTK_TOLERANCE_PCT = 5.0


def detect_gene_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["GENE_NAME", "GENE", "Gene"]:
        if c in df.columns:
            return c
    return None


def _normalize_empty(val: Any) -> tuple[bool, str]:
    """Returns (is_empty, display_str)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True, ""
    s = str(val).strip()
    if s == "" or s.upper() in ("NA", "NAN", "NONE", "NULL", "."):
        return True, ""
    return False, s


def cells_equal_normalized(a: Any, b: Any) -> bool:
    ea, sa = _normalize_empty(a)
    eb, sb = _normalize_empty(b)
    if ea and eb:
        return True
    if ea != eb:
        return False
    return sa == sb


def _try_float(s: str) -> Optional[float]:
    try:
        s = s.strip().replace(",", "")
        return float(s)
    except (ValueError, TypeError, AttributeError):
        return None


def numeric_within_tolerance(cloud_val: Any, linc_val: Any, tolerance_pct: float) -> bool:
    """True if both numeric and On-Prem within ±tolerance_pct of Cloud reference."""
    _, sc = _normalize_empty(cloud_val)
    _, sl = _normalize_empty(linc_val)
    fc = _try_float(sc)
    fl = _try_float(sl)
    if fc is None or fl is None:
        return False
    if abs(fc) < 1e-12:
        return abs(fl) < 1e-12 or abs(fl - fc) < 1e-12
    lo = fc * (1 - tolerance_pct / 100.0)
    hi = fc * (1 + tolerance_pct / 100.0)
    return lo <= fl <= hi


def vec_cells_equal_relaxed(sa: pd.Series, sb: pd.Series) -> pd.Series:
    """Vectorized normalized string equality (NA/empty equivalents match)."""
    sa_str = sa.astype(object).where(sa.notna(), other=np.nan)
    sb_str = sb.astype(object).where(sb.notna(), other=np.nan)
    # string form for comparison
    a_txt = pd.Series(sa_str, index=sa.index, dtype=object).astype(str).str.strip()
    b_txt = pd.Series(sb_str, index=sb.index, dtype=object).astype(str).str.strip()
    na_like = {"", "NAN", "NA", "NONE", "NULL", "."}
    a_empty = sa.isna() | a_txt.str.upper().isin(na_like)
    b_empty = sb.isna() | b_txt.str.upper().isin(na_like)
    both_empty = a_empty & b_empty
    same = a_txt == b_txt
    return both_empty | same


def vec_numeric_tol_ok(sa: pd.Series, sb: pd.Series, tolerance_pct: float) -> pd.Series:
    """Vectorized numeric band check; Cloud (sa) is reference."""
    fa = pd.to_numeric(sa, errors="coerce")
    fb = pd.to_numeric(sb, errors="coerce")
    both_num = fa.notna() & fb.notna()
    lo = fa * (1 - tolerance_pct / 100.0)
    hi = fa * (1 + tolerance_pct / 100.0)
    in_band = both_num & (fb >= lo) & (fb <= hi)
    fz = both_num & fa.abs().lt(1e-12)
    zero_ok = fz & (fb.abs().lt(1e-12) | (fb - fa).abs().lt(1e-12))
    return in_band | zero_ok


def default_strict_columns(shared_cols: list[str]) -> list[str]:
    """Pick clinical-critical columns present in both files (case-sensitive names as in data)."""
    upper_map = {c.upper(): c for c in shared_cols}
    picked = []
    for want in ("CDNA_CHG", "AA_CHG"):
        if want in shared_cols:
            picked.append(want)
        elif want.upper() in upper_map:
            picked.append(upper_map[want.upper()])
    for u, c in upper_map.items():
        if u in ("HGVSG", "HGVS_G", "HGVS.G", "HGVS_GNOMAD") and c not in picked:
            picked.append(c)
    return list(dict.fromkeys(picked))


def build_row_key(df: pd.DataFrame) -> pd.Series:
    for c in KEY_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column for clinical pairing: {c}")
    chrom = df["CHROM"].astype(str).str.strip()
    start = df["START"].astype(str).str.strip()
    ref = df["REF"].astype(str).str.strip()
    alt = df["ALT"].astype(str).str.strip()
    ens = df["ENS_TRANS_ID"].astype(str).str.strip()
    return chrom + "|" + start + "|" + ref + "|" + alt + "|" + ens


def filter_ontarget(df: pd.DataFrame, enabled: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"ontarget_filter": enabled}
    if not enabled:
        return df, meta
    if VARIANT_LOCATION_COL not in df.columns:
        meta["warning"] = f"Column {VARIANT_LOCATION_COL} not found; on-target filter skipped."
        return df, meta
    s = df[VARIANT_LOCATION_COL].astype(str).str.strip().str.upper()
    mask = s == ONTARGET_VALUE.upper()
    return df.loc[mask].copy(), meta


def filter_gene(df: pd.DataFrame, gene: str, gene_col: str) -> pd.DataFrame:
    g = df[gene_col].astype(str).str.strip()
    return df.loc[g == str(gene).strip()].copy()


def run_pairwise_concordance(
    df_cloud: pd.DataFrame,
    df_linc: pd.DataFrame,
    *,
    gene: Optional[str] = None,
    gene_col: Optional[str] = None,
    compare_all_genes: bool = False,
    label_cloud: str = "Cloud",
    label_linc: str = "On-Prem",
    strict_columns: Optional[list[str]] = None,
    compare_columns: Optional[list[str]] = None,
    tolerance_pct: float = 10.0,
    restrict_ontarget: bool = True,
) -> dict[str, Any]:
    """
    Inner-join on CHROM|START|REF|ALT|ENS_TRANS_ID after filters; compare shared columns cell-wise (vectorized).

    If compare_all_genes is True, gene filter is skipped (all rows after on-target filter are compared).
    Otherwise gene and gene_col must be provided.

    If compare_columns is set, only those names (among shared columns) are tested — faster on large files.

    Numeric tolerance: Cloud value is reference; On-Prem must fall within ±tolerance_pct %.
    Strict columns: exact normalized string match only (no numeric tolerance).
    """
    out: dict[str, Any] = {
        "error": None,
        "concordance_pct": None,
        "total_tests": 0,
        "n_failed": 0,
        "n_failed_strict": 0,
        "n_failed_non_strict": 0,
        "n_rows_matched": 0,
        "n_rows_cloud_after_filters": 0,
        "n_rows_linc_after_filters": 0,
        "n_duplicate_rows_dropped_cloud": 0,
        "n_duplicate_rows_dropped_linc": 0,
        "columns_compared": [],
        "strict_columns_used": [],
        "mismatch_df": pd.DataFrame(),
        "mismatch_rows_truncated": False,
        "config_snapshot": {},
    }

    try:
        if not compare_all_genes:
            if not gene or not gene_col:
                out["error"] = "Select a gene, or enable “Compare all genes”."
                return out
            if gene_col not in df_cloud.columns or gene_col not in df_linc.columns:
                out["error"] = f"Gene column {gene_col} missing in one or both files."
                return out

        dc = df_cloud
        dl = df_linc

        meta_c: dict[str, Any] = {}
        meta_l: dict[str, Any] = {}
        if restrict_ontarget:
            dc, meta_c = filter_ontarget(dc, True)
            dl, meta_l = filter_ontarget(dl, True)
        else:
            dc = dc.copy()
            dl = dl.copy()

        if not compare_all_genes:
            dc = filter_gene(dc, gene, gene_col)
            dl = filter_gene(dl, gene, gene_col)

        out["n_rows_cloud_after_filters"] = len(dc)
        out["n_rows_linc_after_filters"] = len(dl)

        n_before_c, n_before_l = len(dc), len(dl)
        # One row per transcript key (same semantics as before: first row wins).
        dc_u = dc.drop_duplicates(subset=KEY_COLS, keep="first")
        dl_u = dl.drop_duplicates(subset=KEY_COLS, keep="first")
        out["n_duplicate_rows_dropped_cloud"] = n_before_c - len(dc_u)
        out["n_duplicate_rows_dropped_linc"] = n_before_l - len(dl_u)

        shared = sorted(set(dc_u.columns) & set(dl_u.columns))
        skip_meta = set(KEY_COLS) | {VARIANT_LOCATION_COL}
        if gene_col:
            skip_meta.add(gene_col)
        compare_cols = [c for c in shared if c not in skip_meta]
        # CONCORDANCE — always exclude EI_TOTAL (case-insensitive)
        compare_cols = [c for c in compare_cols
                        if c not in EXCLUDE_COLS
                        and c.upper() != "EI_TOTAL"]
        if compare_columns:
            want = set(compare_columns)
            compare_cols = [c for c in compare_cols if c in want]
        if not compare_cols:
            out["error"] = "No shared annotation columns to compare (beyond key fields)."
            return out

        # Single inner merge on clinical keys (equivalent to prior index align + intersection).
        merged = dc_u.merge(dl_u, on=KEY_COLS, how="inner", suffixes=("_cloud", "_linc"))
        merged = merged.sort_values(by=KEY_COLS, kind="mergesort").reset_index(drop=True)
        out["n_matched_keys"] = len(merged)
        out["n_rows_matched"] = len(merged)

        if len(merged) == 0:
            out["error"] = "No matching rows after filters (same CHROM, START, REF, ALT, ENS_TRANS_ID)."
            return out

        if strict_columns is None:
            strict_set = set(default_strict_columns(compare_cols))
        else:
            strict_set = set()
            upper_compare = {c.upper(): c for c in compare_cols}
            for sc in strict_columns:
                if sc in compare_cols:
                    strict_set.add(sc)
                elif sc.upper() in upper_compare:
                    strict_set.add(upper_compare[sc.upper()])
        strict_set = {c for c in strict_set if c in compare_cols}
        out["strict_columns_used"] = sorted(strict_set)
        out["columns_compared"] = compare_cols

        n = len(merged)
        n_cols = len(compare_cols)
        total_tests = n * n_cols
        out["total_tests"] = total_tests

        n_fail_strict = 0
        n_fail_other = 0
        mismatch_parts: list[pd.DataFrame] = []
        n_mismatch_collected = 0

        # CONCORDANCE — per-column and per-variant tracking
        col_mismatch_counts: dict[str, int] = {}
        col_total_tests: dict[str, int] = {}
        col_concordance_pct: dict[str, float] = {}
        fail_masks_per_col: list[pd.Series] = []
        _vartk_upper = {c.upper() for c in VARTK_COLS}

        # CONCORDANCE — build row-level composite key for variant tracking
        _merged_keys_str = (
            merged["CHROM"].astype(str) + ":"
            + merged["START"].astype(str) + ":"
            + merged["REF"].astype(str) + ":"
            + merged["ALT"].astype(str) + ":"
            + merged["ENS_TRANS_ID"].astype(str)
        )

        for col in compare_cols:
            sa = merged[f"{col}_cloud"]
            sb = merged[f"{col}_linc"]
            is_strict = col in strict_set
            # CONCORDANCE — VarTk ±5% takes priority, then strict, then default tolerance
            is_vartk = col.upper() in _vartk_upper
            if is_vartk:
                ok = vec_cells_equal_relaxed(sa, sb) | vec_numeric_tol_ok(sa, sb, VARTK_TOLERANCE_PCT)
            elif is_strict:
                ok = vec_cells_equal_relaxed(sa, sb)
            else:
                ok = vec_cells_equal_relaxed(sa, sb) | vec_numeric_tol_ok(sa, sb, tolerance_pct)
            fail = ~ok
            n_fail = int(fail.sum())
            if is_strict:
                n_fail_strict += n_fail
            else:
                n_fail_other += n_fail

            # CONCORDANCE — per-column stats
            col_mismatch_counts[col] = n_fail
            col_total_tests[col] = n
            col_concordance_pct[col] = round(100.0 * (n - n_fail) / n, 4) if n else 0.0
            fail_masks_per_col.append(fail.rename(col))

            if n_fail > 0 and n_mismatch_collected < MAX_MISMATCH_ROWS:
                bad = merged.loc[fail]
                take = min(len(bad), MAX_MISMATCH_ROWS - n_mismatch_collected)
                bad = bad.iloc[:take]
                sub_a = bad[f"{col}_cloud"]
                sub_b = bad[f"{col}_linc"]
                bad_keys = bad[KEY_COLS].astype(str)
                composite = (
                    bad_keys["CHROM"]
                    + ":"
                    + bad_keys["START"]
                    + ":"
                    + bad_keys["REF"]
                    + ":"
                    + bad_keys["ALT"]
                    + ":"
                    + bad_keys["ENS_TRANS_ID"]
                )
                cloud_vals = sub_a.fillna("").astype(str).str.strip().values
                linc_vals = sub_b.fillna("").astype(str).str.strip().values
                part_df = bad_keys.copy()
                part_df["variant_key"] = composite.values
                part_df["column"] = col
                part_df[f"{label_cloud}_value"] = cloud_vals
                part_df[f"{label_linc}_value"] = linc_vals
                part_df["strict_column"] = is_strict
                part_df["label"] = (composite + ":" + str(col)).values
                mismatch_parts.append(part_df)
                n_mismatch_collected += len(part_df)

        n_failed = n_fail_strict + n_fail_other
        out["n_failed"] = n_failed
        out["n_failed_strict"] = n_fail_strict
        out["n_failed_non_strict"] = n_fail_other
        out["concordance_pct"] = round(100.0 * (total_tests - n_failed) / total_tests, 4) if total_tests else 0.0
        out["mismatch_rows_truncated"] = n_failed > n_mismatch_collected
        if mismatch_parts:
            out["mismatch_df"] = pd.concat(mismatch_parts, ignore_index=True)

        # CONCORDANCE — match_df: rows where every compared column matched perfectly
        if fail_masks_per_col:
            any_fail = pd.concat(fail_masks_per_col, axis=1).any(axis=1)
        else:
            any_fail = pd.Series(False, index=merged.index)
        perfect_match_rows = merged[~any_fail][KEY_COLS].copy()
        perfect_match_rows["columns_checked"] = len(compare_cols)
        match_df = perfect_match_rows.head(MAX_MISMATCH_ROWS).reset_index(drop=True)

        # CONCORDANCE — per-variant mismatch summary
        variant_mismatch_counts: dict[str, int] = {}
        strict_fail_variants: list[str] = []
        if fail_masks_per_col:
            fail_matrix = pd.concat(fail_masks_per_col, axis=1)
            cols_failed_per_row = fail_matrix.sum(axis=1)
            rows_with_fail = cols_failed_per_row[cols_failed_per_row > 0]
            for idx in rows_with_fail.index:
                vk = _merged_keys_str.iloc[idx] if idx < len(_merged_keys_str) else str(idx)
                variant_mismatch_counts[vk] = int(rows_with_fail[idx])
            # strict-column fail variants
            strict_col_names = [col for col in compare_cols if col in strict_set]
            if strict_col_names:
                strict_fail_cols = [s for s in fail_masks_per_col if s.name in strict_set]
                if strict_fail_cols:
                    strict_any_fail = pd.concat(strict_fail_cols, axis=1).any(axis=1)
                    for idx in strict_any_fail[strict_any_fail].index:
                        vk = _merged_keys_str.iloc[idx] if idx < len(_merged_keys_str) else str(idx)
                        strict_fail_variants.append(vk)

        # CONCORDANCE — enhanced return keys
        out["col_mismatch_counts"] = col_mismatch_counts
        out["col_total_tests"] = col_total_tests
        out["col_concordance_pct"] = col_concordance_pct
        out["variant_mismatch_counts"] = variant_mismatch_counts
        out["strict_fail_variants"] = strict_fail_variants[:1000]
        out["match_df"] = match_df
        out["n_perfect_match_rows"] = len(perfect_match_rows)
        out["n_mismatch_rows"] = int(any_fail.sum())
        out["excluded_cols"] = sorted(EXCLUDE_COLS)

        out["config_snapshot"] = {
            "gene": gene if not compare_all_genes else None,
            "gene_col": gene_col,
            "compare_all_genes": compare_all_genes,
            "label_cloud": label_cloud,
            "label_linc": label_linc,
            "tolerance_pct": tolerance_pct,
            "restrict_ontarget": restrict_ontarget,
            "compare_columns_filter": bool(compare_columns),
            "meta_cloud": meta_c,
            "meta_linc": meta_l,
            "max_mismatch_rows_stored": MAX_MISMATCH_ROWS,
        }
        if meta_c.get("warning"):
            out["warnings"] = [meta_c["warning"]]
        if meta_l.get("warning") and meta_l.get("warning") != meta_c.get("warning"):
            out.setdefault("warnings", []).append(meta_l["warning"])

    except Exception as e:
        out["error"] = str(e)
    return out


def validation_summary_line(result: dict[str, Any]) -> str:
    """One-line factual summary for PDF footer."""
    if result.get("error"):
        return f"Validation incomplete: {result['error']}"
    if result.get("total_tests", 0) == 0:
        return "No cell-level tests were executed."
    fs = result.get("n_failed_strict", 0)
    if fs > 0:
        return (
            f"Concordance {result.get('concordance_pct', 0):.2f}% with {fs} strict (clinical) field mismatch(es); review required."
        )
    return (
        f"Concordance {result.get('concordance_pct', 0):.2f}%; no strict clinical field mismatches in the selected scope."
    )
