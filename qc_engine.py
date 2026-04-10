"""
VariMAT QC Tool - QC engine: metrics, comparison, duplicate analysis, headers, overlap, mismatches.
"""

from typing import Any

import numpy as np
import pandas as pd

from utils import REQUIRED_COLS, create_variant_key


def compute_record_metrics(
    dataframes: dict[str, pd.DataFrame],
    variant_keys: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Record-level metrics per file: total rows, duplicate rows, file size proxy.
    """
    rows = []
    for name, df in dataframes.items():
        n = len(df)
        keys = variant_keys[name]
        n_unique = keys.nunique()
        n_dup_rows = n - n_unique
        # Approximate size in MB (object dtype rough)
        size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        rows.append({
            "File": name,
            "Total Rows": n,
            "Unique Variants": n_unique,
            "Duplicate Rows": n_dup_rows,
            "Size (MB)": round(size_mb, 2),
        })
    return pd.DataFrame(rows)


def compute_unique_variant_metrics(
    dataframes: dict[str, pd.DataFrame],
    variant_keys: dict[str, pd.Series],
) -> dict[str, Any]:
    """
    Unique variant-level metrics: per-file unique counts, shared variants, overlap.
    """
    sets = {name: set(keys.dropna().unique()) for name, keys in variant_keys.items()}
    names = list(sets.keys())
    n_files = len(names)

    all_variants = set()
    for s in sets.values():
        all_variants |= s

    common_all = all_variants
    for s in sets.values():
        common_all = common_all & s

    per_file = {}
    for name in names:
        s = sets[name]
        others = set()
        for n, x in sets.items():
            if n != name:
                others |= x
        only_in_this = s - others
        per_file[name] = {
            "unique_variants": len(s),
            "only_in_this_file": len(only_in_this),
        }

    # Overlap matrix: for each pair, shared count
    overlap = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                overlap[(a, b)] = len(sets[a] & sets[b])

    return {
        "names": names,
        "sets": sets,
        "total_unique_across_all": len(all_variants),
        "common_in_all": len(common_all),
        "common_variant_ids": common_all,
        "per_file": per_file,
        "overlap_pairs": overlap,
    }


def duplicate_variant_analysis(
    variant_keys: dict[str, pd.Series],
) -> dict[str, Any]:
    """
    Duplicate variant analysis: frequency per file, top duplicated, histogram data.
    """
    result = {}
    for name, keys in variant_keys.items():
        vc = keys.value_counts()
        dup = vc[vc > 1]
        result[name] = {
            "variant_counts": vc,
            "duplicated_variants": dup,
            "n_duplicated_variants": len(dup),
            "total_duplicate_entries": (vc - 1).clip(0).sum(),
            "top_duplicated": dup.head(20) if len(dup) else pd.Series(dtype=float),
            "histogram_counts": vc.value_counts().sort_index() if len(vc) else pd.Series(dtype=float),
        }
    return result


def compare_headers(dataframes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Compare headers across files: missing/extra columns, similarity score.
    """
    names = list(dataframes.keys())
    all_cols = set()
    for df in dataframes.values():
        all_cols.update(df.columns.tolist())
    all_cols = sorted(all_cols)

    # Build matrix: column -> which files have it
    col_present = {col: [] for col in all_cols}
    for col in all_cols:
        for n in names:
            col_present[col].append(col in dataframes[n].columns)

    table = []
    for col in all_cols:
        row = {"Column": col}
        for i, n in enumerate(names):
            row[n] = "Yes" if col_present[col][i] else "No"
        table.append(row)

    # Similarity: pairwise Jaccard
    header_sets = [set(df.columns) for df in dataframes.values()]
    n_f = len(header_sets)
    scores = []
    for i in range(n_f):
        for j in range(i + 1, n_f):
            inter = len(header_sets[i] & header_sets[j])
            union = len(header_sets[i] | header_sets[j])
            scores.append(inter / union if union else 1.0)
    header_similarity = np.mean(scores) * 100 if scores else 100.0

    return {
        "column_table": pd.DataFrame(table),
        "all_columns": all_cols,
        "header_similarity_pct": round(header_similarity, 2),
        "file_names": names,
        "has_mismatch": any(s < 100 for s in [np.mean([col_present[c][i] for c in all_cols]) * 100 for i in range(n_f)]),
    }


def variant_overlap_analysis(
    variant_keys: dict[str, pd.Series],
) -> dict[str, Any]:
    """
    Variant overlap: common in all, unique per file, missing between files.
    """
    names = list(variant_keys.keys())
    sets = {n: set(v.dropna().unique()) for n, v in variant_keys.items()}

    common_all = None
    for s in sets.values():
        if common_all is None:
            common_all = s.copy()
        else:
            common_all &= s

    unique_per_file = {}
    for n in names:
        others = set()
        for k, v in sets.items():
            if k != n:
                others |= v
        unique_per_file[n] = sets[n] - others

    # Missing between pairs: in A but not in B
    missing_pairs = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                missing_pairs[(a, b)] = sets[a] - sets[b]

    total_union = set()
    for s in sets.values():
        total_union |= s

    return {
        "common_in_all": common_all or set(),
        "common_count": len(common_all) if common_all else 0,
        "unique_per_file": unique_per_file,
        "missing_pairs": missing_pairs,
        "sets": sets,
        "names": names,
        "total_union": total_union,
    }


def column_mismatch_detection(
    dataframes: dict[str, pd.DataFrame],
    variant_keys: dict[str, pd.Series],
    common_variant_ids: set,
) -> dict[str, Any]:
    """
    For variants present in multiple files, compare values in shared columns.
    Detect annotation/classification/gene/transcript differences.
    """
    # PERF: vectorized join per file-pair vs row-wise Python loops over common variants.
    names = list(dataframes.keys())
    shared_cols = set(dataframes[names[0]].columns)
    for df in dataframes.values():
        shared_cols &= set(df.columns)
    compare_cols = [
        c
        for c in shared_cols
        if c not in REQUIRED_COLS
        and all(c in df.columns for df in dataframes.values())
    ]

    keyed: dict[str, pd.DataFrame] = {}
    for n in names:
        tmp = dataframes[n].copy(deep=False)
        tmp["_vk"] = variant_keys[n]
        tmp = tmp[tmp["_vk"].isin(common_variant_ids)]
        tmp = tmp.drop_duplicates(subset=["_vk"], keep="first")
        keyed[n] = tmp.set_index("_vk")

    ref_name = names[0]
    ref_df = keyed[ref_name][compare_cols]

    mismatches: list[pd.DataFrame] = []
    for other_name in names[1:]:
        other_df = keyed[other_name][compare_cols]
        joined = ref_df.join(other_df, how="inner", lsuffix="_ref", rsuffix="_other")
        for col in compare_cols:
            a = joined[f"{col}_ref"].fillna("").astype(str).str.strip()
            b = joined[f"{col}_other"].fillna("").astype(str).str.strip()
            mask = a != b
            if mask.any():
                bad = joined[mask].index
                sub = pd.DataFrame(
                    {
                        "Variant ID": bad,
                        "Column": col,
                        f"{ref_name} value": a[mask].values,
                        f"{other_name} value": b[mask].values,
                    }
                )
                mismatches.append(sub)

    mismatch_df = (
        pd.concat(mismatches, ignore_index=True).head(50_000) if mismatches else pd.DataFrame()
    )
    return {
        "mismatch_table": mismatch_df,
        "mismatch_count": len(mismatch_df),
        "compared_columns": compare_cols,
        "common_variants_checked": min(len(common_variant_ids), len(ref_df)),
    }


def variant_consistency_analysis(
    dataframes: dict[str, pd.DataFrame],
    variant_keys: dict[str, pd.Series],
    common_variant_ids: set,
) -> dict[str, Any]:
    """
    Group by variant_key and compare annotations across files (e.g. transcript-level differences).
    """
    # PERF: concat + groupby nunique per column instead of Python loops over variant ids.
    names = list(dataframes.keys())
    shared_cols = set(dataframes[names[0]].columns)
    for df in dataframes.values():
        shared_cols &= set(df.columns)
    compare_cols = [c for c in shared_cols if c not in REQUIRED_COLS]

    frames = []
    for n in names:
        tmp = dataframes[n].copy(deep=False)
        tmp["_vk"] = variant_keys[n]
        tmp["_file"] = n
        tmp = tmp[tmp["_vk"].isin(common_variant_ids)]
        frames.append(tmp[["_vk", "_file"] + compare_cols])

    combined = pd.concat(frames, ignore_index=True)

    inconsistencies: list[dict[str, Any]] = []
    for col in compare_cols:
        g = combined.groupby("_vk")[col].agg(
            lambda x: x.dropna().astype(str).str.strip().nunique()
        )
        bad_vks = g[g > 1].index
        if len(bad_vks):
            for vk in bad_vks[:3000]:
                inconsistencies.append(
                    {
                        "Variant ID": vk,
                        "Column": col,
                        "Distinct values": int(g.loc[vk]),
                    }
                )
            break  # one column per variant max (matches original "break" logic)

    return {
        "inconsistency_df": pd.DataFrame(inconsistencies) if inconsistencies else pd.DataFrame(),
        "inconsistency_count": len(inconsistencies),
    }


def missing_data_analysis(dataframes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Per-file and per-column NA percentage, empty columns, missing data heatmap data.
    """
    names = list(dataframes.keys())
    all_cols = set()
    for df in dataframes.values():
        all_cols.update(df.columns)
    all_cols = sorted(all_cols)

    na_pct = {}
    for n in names:
        df = dataframes[n]
        nrows = len(df)
        if nrows == 0:
            na_pct[n] = pd.Series(100.0, index=all_cols).reindex(all_cols).fillna(100)
        else:
            na_pct[n] = (df.isna().sum() / nrows * 100).reindex(all_cols).fillna(100)

    heatmap_data = pd.DataFrame(na_pct)  # columns = file names, index = all_cols (data column names)
    empty_cols = [c for c in all_cols if c in heatmap_data.index and (heatmap_data.loc[c] >= 100).all()]
    high_na = [c for c in all_cols if c in heatmap_data.index and heatmap_data.loc[c].max() >= 50]

    return {
        "na_pct_df": heatmap_data,
        "empty_columns": empty_cols,
        "high_na_columns": high_na,
        "file_names": names,
        "all_columns": all_cols,
    }


def generate_qc_score(
    header_similarity: float,
    common_count: int,
    total_unique: int,
    mismatch_count: int,
    inconsistency_count: int,
    high_na_count: int,
    duplicate_pattern_penalty: float,
) -> tuple[int, str]:
    """
    QC score 0-100 and status label.
    """
    score = 100.0
    if header_similarity < 100:
        score -= 10
    if total_unique > 0 and common_count / total_unique < 0.5:
        score -= 15
    if mismatch_count > 100:
        score -= 20
    elif mismatch_count > 0:
        score -= 10
    if high_na_count > 10:
        score -= 10
    if inconsistency_count > 50:
        score -= 15
    elif inconsistency_count > 0:
        score -= 5
    score -= duplicate_pattern_penalty  # up to 10
    score = max(0, min(100, round(score)))

    if score >= 90:
        status = "Excellent"
    elif score >= 75:
        status = "Good"
    elif score >= 50:
        status = "Fair"
    else:
        status = "Needs attention"
    return int(score), status
