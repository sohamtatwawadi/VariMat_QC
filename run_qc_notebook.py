"""
VariMAT QC — Run full QC from Jupyter or command line (no Streamlit UI).
Uses the same logic as the dashboard but faster: load from paths, run QC, print summary, export CSV/PDF.

Jupyter usage:
    from run_qc_notebook import run_qc, FILE_PATHS, OUTPUT_DIR

    # Option 1: set paths in the module then run
    FILE_PATHS[:] = ["/path/to/file1.txt", "/path/to/file2.txt"]
    results = run_qc(FILE_PATHS, output_dir="./qc_output")

    # Option 2: pass paths directly
    results = run_qc(["/path/to/a.txt", "/path/to/b.txt"], output_dir="./qc_output")

    # results is a dict with dataframes, overlap_data, qc_score, etc. for further analysis.
    # The script adds varimat_qc_tool to sys.path automatically so imports work from any cwd.

Command line:
    python run_qc_notebook.py /path/to/file1.txt /path/to/file2.txt
"""

import os
import sys

# Ensure varimat_qc_tool is on path so qc_engine, utils, report_generator can be imported
# (needed when running from Jupyter with cwd = project root or elsewhere)
if "__file__" in globals():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
else:
    _cwd = os.getcwd()
    for _d in [os.path.join(_cwd, "varimat_qc_tool"), _cwd]:
        if os.path.isfile(os.path.join(_d, "qc_engine.py")) and _d not in sys.path:
            sys.path.insert(0, _d)
            break

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from qc_engine import (
    compare_headers,
    compute_record_metrics,
    compute_unique_variant_metrics,
    duplicate_variant_analysis,
    generate_qc_score,
    missing_data_analysis,
    variant_consistency_analysis,
    variant_overlap_analysis,
    column_mismatch_detection,
)
from report_generator import generate_csv_exports, generate_pdf_report, get_pdf_filename
from utils import create_variant_key, safe_load_varimat_from_path

# ---------------------------------------------------------------------------
# Configure paths here (for Jupyter: edit this list and run the cell)
# Or pass paths as command-line args: python run_qc_notebook.py /path/to/file1.txt /path/to/file2.txt
# ---------------------------------------------------------------------------
FILE_PATHS = [
    "/Users/sohamtatwawadi/cloud_content_check/395_linc_v1_GATK.VariMAT2.6.1_grch38_VarMiner2_filt.txt",
    "/Users/sohamtatwawadi/cloud_content_check/9708896.v5_9.GATK.VariMAT2.6.1_grch38_VarMiner2_filt.txt",
]

# Optional: directory to write CSV and PDF exports (None = skip export)
OUTPUT_DIR = "/Users/sohamtatwawadi/cloud_content_check/"


def run_full_qc_standalone(dataframes: dict) -> dict:
    """Same QC logic as the Streamlit app, no UI/cache."""
    if not dataframes or len(dataframes) < 2:
        return None
    try:
        variant_keys = {name: create_variant_key(df) for name, df in dataframes.items()}
    except Exception as e:
        return {"error": str(e)}

    try:
        record_metrics = compute_record_metrics(dataframes, variant_keys)
        unique_metrics = compute_unique_variant_metrics(dataframes, variant_keys)
        dup_analysis = duplicate_variant_analysis(variant_keys)
        header_result = compare_headers(dataframes)
        overlap_data = variant_overlap_analysis(variant_keys)
        common_ids = unique_metrics.get("common_variant_ids", set())
        mismatch_result = column_mismatch_detection(dataframes, variant_keys, common_ids)
        consistency_result = variant_consistency_analysis(dataframes, variant_keys, common_ids)
        missing_data = missing_data_analysis(dataframes)
    except Exception as e:
        return {"error": str(e)}

    total_records = sum(len(df) for df in dataframes.values())
    total_unique = unique_metrics.get("total_unique_across_all", 0)
    common_count = unique_metrics.get("common_in_all", 0)
    dup_penalty = 0.0
    if dup_analysis:
        dup_counts = [d.get("n_duplicated_variants", 0) for d in dup_analysis.values()]
        if dup_counts and len(set(dup_counts)) > 1 and max(dup_counts) - min(dup_counts) > 100:
            dup_penalty = 10.0
    try:
        qc_score, qc_status = generate_qc_score(
            header_similarity=header_result.get("header_similarity_pct", 0),
            common_count=common_count,
            total_unique=total_unique,
            mismatch_count=mismatch_result.get("mismatch_count", 0),
            inconsistency_count=consistency_result.get("inconsistency_count", 0),
            high_na_count=len(missing_data.get("high_na_columns", [])),
            duplicate_pattern_penalty=dup_penalty,
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "dataframes": dataframes,
        "variant_keys": variant_keys,
        "record_metrics": record_metrics,
        "unique_metrics": unique_metrics,
        "dup_analysis": dup_analysis,
        "header_result": header_result,
        "overlap_data": overlap_data,
        "mismatch_result": mismatch_result,
        "consistency_result": consistency_result,
        "missing_data": missing_data,
        "qc_score": qc_score,
        "qc_status": qc_status,
        "total_records": total_records,
        "total_unique": total_unique,
        "common_count": common_count,
        "n_files": len(dataframes),
    }


def load_files_from_paths(paths: list):
    """Load VariMAT files from disk in parallel. Returns (dataframes_dict, list of error messages)."""
    paths = [os.path.expanduser(p.strip()) for p in paths if p and p.strip()][:10]
    valid = [p for p in paths if os.path.isfile(p)]
    errors = []
    for p in paths:
        if not os.path.isfile(p):
            errors.append(f"Not found: {p}")

    if len(valid) < 2:
        return {}, errors

    def load_one(path: str):
        name = os.path.basename(path)
        df, err = safe_load_varimat_from_path(path)
        return (name, df, err)

    dataframes = {}
    max_workers = min(len(valid), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(load_one, p) for p in valid]
        for f in as_completed(futures):
            name, df, err = f.result()
            if err or df is None:
                errors.append(f"{name}: {err or 'Load failed'}")
            else:
                dataframes[name] = df
    return dataframes, errors


def print_summary(results: dict) -> None:
    """Print key metrics and QC score to stdout."""
    if not results or results.get("error"):
        print("QC failed:", results.get("error", "No results"))
        return
    r = results
    n = r.get("n_files", 0)
    total = r.get("total_records", 0)
    unique = r.get("total_unique", 0)
    common = r.get("common_count", 0)
    score = r.get("qc_score", 0)
    status = r.get("qc_status", "?")
    header_pct = r.get("header_result", {}).get("header_similarity_pct", 0)
    print("\n" + "=" * 60)
    print("VariMAT QC — Summary")
    print("=" * 60)
    print(f"  Files compared:        {n}")
    print(f"  Total records:        {total:,}")
    print(f"  Unique variants:      {unique:,}")
    print(f"  Common in all files:  {common:,}")
    print(f"  Header similarity:    {header_pct}%")
    print(f"  QC Score:             {score}/100 ({status})")
    print("=" * 60 + "\n")
    if r.get("record_metrics") is not None and not r["record_metrics"].empty:
        print("Record-level metrics:")
        print(r["record_metrics"].to_string(index=False))
        print()


def export_results(results: dict, output_dir: str) -> None:
    """Write CSV exports and PDF report to output_dir."""
    if not results or results.get("error") or not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    r = results
    qc_summary = {
        "files_compared": r.get("n_files", 0),
        "total_records": r.get("total_records", 0),
        "unique_variants": r.get("total_unique", 0),
        "common_in_all": r.get("common_count", 0),
        "qc_score": r.get("qc_score", 0),
        "qc_status": r.get("qc_status", ""),
        "header_similarity_pct": r.get("header_result", {}).get("header_similarity_pct", 0),
    }
    csvs = generate_csv_exports(
        r.get("record_metrics"),
        r.get("overlap_data") or {},
        r.get("mismatch_result", {}).get("mismatch_table"),
        r.get("dup_analysis") or {},
        r.get("header_result", {}).get("column_table"),
        r.get("overlap_data", {}).get("unique_per_file", {}),
        r.get("overlap_data", {}).get("missing_pairs", {}),
        qc_summary=qc_summary,
        clinical_mismatch_df=r.get("clinical_mismatch_df"),
    )
    for fname, content in (csvs or {}).items():
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as out:
            out.write(content)
        print(f"  Wrote {path}")
    pdf_bytes = generate_pdf_report(
        qc_score=r.get("qc_score", 0),
        qc_status=r.get("qc_status", ""),
        n_files=r.get("n_files", 0),
        total_records=r.get("total_records", 0),
        total_unique=r.get("total_unique", 0),
        common_count=r.get("common_count", 0),
        record_metrics=r.get("record_metrics"),
        header_similarity=r.get("header_result", {}).get("header_similarity_pct", 0),
        header_table=r.get("header_result", {}).get("column_table"),
        overlap_data=r.get("overlap_data") or {},
        mismatch_count=r.get("mismatch_result", {}).get("mismatch_count", 0),
        dup_analysis=r.get("dup_analysis") or {},
        management_summary={"variant_consistency": "High" if r.get("qc_score", 0) >= 75 else "Medium" if r.get("qc_score", 0) >= 50 else "Low"},
        clinical_concordance=r.get("clinical_concordance"),
    )
    pdf_path = os.path.join(output_dir, get_pdf_filename())
    with open(pdf_path, "wb") as out:
        out.write(pdf_bytes)
    print(f"  Wrote {pdf_path}")


def run_qc(paths: list, output_dir: str = None, verbose: bool = True) -> dict:
    """
    Load files from paths, run full QC, print summary, optionally export to output_dir.
    Returns the full results dict (for use in Jupyter to inspect dataframes, overlap_data, etc.).
    """
    if verbose:
        print("Loading files from disk...")
    t0 = time.perf_counter()
    dataframes, load_errors = load_files_from_paths(paths)
    for e in load_errors:
        print(f"  Error: {e}")
    if len(dataframes) < 2:
        print("Need at least 2 valid files. Aborting.")
        return None
    if verbose:
        print(f"  Loaded {len(dataframes)} files in {time.perf_counter() - t0:.1f}s")

    if verbose:
        print("Running full QC...")
    t1 = time.perf_counter()
    results = run_full_qc_standalone(dataframes)
    if verbose:
        print(f"  QC completed in {time.perf_counter() - t1:.1f}s")

    if results is None or results.get("error"):
        print("QC failed:", results.get("error", "Unknown"))
        return results

    print_summary(results)
    if output_dir:
        if verbose:
            print(f"Exporting to {output_dir}...")
        export_results(results, output_dir)

    if verbose:
        print(f"Total time: {time.perf_counter() - t0:.1f}s")
    return results


def main():
    paths = FILE_PATHS
    if len(sys.argv) >= 3:
        paths = sys.argv[1:]
    if not paths or len(paths) < 2:
        print("Usage: python run_qc_notebook.py <path1> <path2> [path3 ...]")
        print("Or set FILE_PATHS in the script and run.")
        return
    run_qc(paths, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
