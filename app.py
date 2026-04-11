"""
VariMAT QC Dashboard - Production-ready QC tool for genomics variant files.
Upload multiple VariMAT files, run QC, compare variants, and download reports.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit import fragment as st_fragment
except ImportError:

    def st_fragment(f):
        return f


def _qc_local_paths_signature(paths):
    """Canonical (order-independent) signature for on-disk paths for QC cache."""
    rows = []
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        ap = os.path.abspath(os.path.expanduser(p.strip()))
        try:
            rows.append((ap, os.path.getmtime(ap), os.path.getsize(ap)))
        except OSError:
            continue
    return tuple(sorted(rows, key=lambda x: x[0])) if len(rows) >= 2 else None


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
from report_generator import (
    generate_csv_exports,
    generate_pdf_report,
    get_pdf_filename,
    generate_concordance_pdf_report,
    generate_concordance_csv_exports,
    get_concordance_pdf_filename,
)
from utils import create_variant_key, load_varimat_path_worker

# Page config
st.set_page_config(
    page_title="VariMAT QC Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern light theme — professional UI/UX
st.markdown("""
<style>
    /* Base: light background and typography */
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%); }
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    h1, h2, h3 { color: #1e293b !important; font-weight: 600 !important; letter-spacing: -0.02em; }
    p, .stMarkdown { color: #475569 !important; }

    /* KPI cards: clean white cards with subtle shadow and teal accent */
    .kpi-card {
        background: #ffffff;
        color: #1e293b;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
        margin: 0.35rem 0;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0f766e;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.06); }
    .kpi-card h3 { color: #64748b !important; font-size: 0.8rem !important; font-weight: 600 !important; margin: 0 0 0.35rem 0 !important; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-card .value { font-size: 1.65rem; font-weight: 700; color: #0f766e; letter-spacing: -0.02em; }

    /* Tabs: modern pill style */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: #f1f5f9; padding: 6px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #ffffff !important; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }

    /* QC score badges */
    .qc-score-excellent { color: #059669 !important; font-weight: 700; }
    .qc-score-good { color: #0d9488 !important; font-weight: 700; }
    .qc-score-fair { color: #d97706 !important; font-weight: 700; }
    .qc-score-poor { color: #dc2626 !important; font-weight: 700; }

    /* Alert / info boxes */
    .warning-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 14px 16px; border-radius: 8px; margin: 12px 0; color: #92400e; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .success-box { background: #ecfdf5; border-left: 4px solid #10b981; padding: 14px 16px; border-radius: 8px; margin: 12px 0; color: #065f46; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .stMarkdown { color: #475569 !important; }

    /* Dataframes and inputs */
    .stDataFrame { border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; }
    div[data-testid="stExpander"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }

    /* Page header: professional banner */
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 118, 110, 0.2);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .main-header h1 { color: white !important; margin: 0 !important; font-size: 1.65rem !important; }
    .main-header p { color: rgba(255,255,255,0.92) !important; margin: 0.35rem 0 0 0 !important; font-size: 0.9rem !important; }

    /* Executive validation panel */
    .validation-panel {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 4px 20px rgba(15, 118, 110, 0.08);
    }
    .validation-panel h3 { color: #0f766e !important; font-size: 1.05rem !important; margin-bottom: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)


def _genes_both_for_clinical(df_cloud: pd.DataFrame, df_linc: pd.DataFrame, gene_col: str, restrict_ontarget: bool) -> list:
    from clinical_concordance import filter_ontarget

    dc, _ = filter_ontarget(df_cloud.copy(), restrict_ontarget)
    dl, _ = filter_ontarget(df_linc.copy(), restrict_ontarget)
    if gene_col not in dc.columns or gene_col not in dl.columns:
        return []
    gc = set(dc[gene_col].dropna().astype(str).str.strip())
    gl = set(dl[gene_col].dropna().astype(str).str.strip())
    return sorted(gc & gl)[:2000]


def _render_executive_validation_cloud_linc(dataframes: dict) -> None:
    """Cloud vs On-Prem cell-level concordance; stores result in session_state for PDF."""
    from clinical_concordance import (
        detect_gene_column,
        default_strict_columns,
        run_pairwise_concordance,
    )

    if not dataframes or len(dataframes) < 2:
        return

    names = list(dataframes.keys())
    with st.expander("Executive validation — Cloud vs On-Prem", expanded=False):
        st.markdown(
            '<p style="color:#64748b;font-size:0.9rem;">Transcript-level match: <code>CHROM</code>, <code>START</code>, '
            "<code>REF</code>, <code>ALT</code>, <code>ENS_TRANS_ID</code>. Optional on-target filter "
            "(<code>VARIANT_LOCATION</code> = ONTARGET). Cloud is the numeric reference for tolerance. On-Prem is compared against Cloud. "
            "Comparisons are <b>vectorized</b> for speed; use <b>local paths</b> to avoid long upload times. "
            "For multi‑GB files, prefer <b>one gene</b> or <b>limit columns</b> so total runtime stays within a few minutes.</p>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            cloud_name = st.selectbox("Cloud (reference) file", names, key="ev_cloud_file")
        linc_candidates = [n for n in names if n != cloud_name]
        with c2:
            if not linc_candidates:
                st.warning("Need at least two distinct files for Cloud vs On-Prem.")
                return
            linc_name = st.selectbox("On-Prem file", linc_candidates, key="ev_linc_file")

        df_c = dataframes[cloud_name]
        df_l = dataframes[linc_name]

        compare_all = st.checkbox(
            "Compare all genes (skip gene filter)",
            value=False,
            key="ev_compare_all_genes",
            help="Uses all rows after the on-target filter. Much larger than a single gene — limit columns if runtime is high.",
        )

        gene_col_c = detect_gene_column(df_c)
        gene_col_l = detect_gene_column(df_l)
        gene_col = None
        if not compare_all:
            if not gene_col_c or gene_col_c != gene_col_l:
                st.warning("Both files need the same gene column (GENE_NAME / GENE / Gene), or enable “Compare all genes”.")
                return
            gene_col = gene_col_c
        else:
            if gene_col_c and gene_col_c == gene_col_l:
                gene_col = gene_col_c

        restrict_ot = st.checkbox("Restrict to on-target only (VARIANT_LOCATION)", value=True, key="ev_ontarget")
        tol = st.slider("Numeric tolerance (± %, On-Prem vs Cloud)", min_value=0.0, max_value=20.0, value=10.0, step=1.0, key="ev_tol")

        gene_sel = None
        if not compare_all:
            genes = _genes_both_for_clinical(df_c, df_l, gene_col, restrict_ot)
            if not genes:
                st.info("No overlapping gene symbols after filters; adjust files, disable on-target filter, or use “Compare all genes”.")
                return
            gene_sel = st.selectbox("Gene (one)", genes, key="ev_gene")

        shared_cols = sorted(set(df_c.columns) & set(df_l.columns))
        def_strict = default_strict_columns(shared_cols)
        strict_sel = st.multiselect(
            "Strict columns (zero tolerance — clinical report fields)",
            options=shared_cols,
            default=[c for c in def_strict if c in shared_cols],
            key="ev_strict",
        )
        col_subset = st.multiselect(
            "Optional: columns to compare (empty = all shared except keys — limiting columns speeds up large files)",
            options=[c for c in shared_cols if c not in {"CHROM", "START", "REF", "ALT", "ENS_TRANS_ID"}],
            default=[],
            key="ev_col_subset",
            help="Leave empty to compare every shared annotation column. Selecting a subset (e.g. 30–80 fields) keeps runs fast.",
        )

        if st.button("Run Cloud vs On-Prem validation", type="primary", key="ev_run"):
            with st.spinner("Running vectorized transcript-level concordance (may take 1–3 min on very large files)…"):
                try:
                    res = run_pairwise_concordance(
                        df_c,
                        df_l,
                        gene=gene_sel,
                        gene_col=gene_col,
                        compare_all_genes=compare_all,
                        label_cloud="Cloud",
                        label_linc="On-Prem",
                        strict_columns=strict_sel if strict_sel else None,
                        compare_columns=col_subset if col_subset else None,
                        tolerance_pct=float(tol),
                        restrict_ontarget=restrict_ot,
                    )
                    st.session_state["clinical_concordance"] = res
                except Exception as e:
                    st.session_state["clinical_concordance"] = {"error": str(e), "mismatch_df": pd.DataFrame()}

        res = st.session_state.get("clinical_concordance")
        if res:
            if res.get("error") and not res.get("total_tests"):
                st.error(res["error"])
            elif res.get("error") and res.get("total_tests"):
                st.warning(res["error"])
            for w in res.get("warnings") or []:
                st.warning(w)

            if res.get("total_tests"):
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Concordance %", f"{res.get('concordance_pct', 0):.2f}")
                with m2:
                    st.metric("Cell-level tests", f"{res.get('total_tests', 0):,}")
                with m3:
                    st.metric("Strict mismatches", res.get("n_failed_strict", 0))
                with m4:
                    st.metric("Matched rows", res.get("n_rows_matched", 0))

                st.caption(
                    f"Rows after filters — Cloud: {res.get('n_rows_cloud_after_filters', 0):,} · "
                    f"On-Prem: {res.get('n_rows_linc_after_filters', 0):,} · "
                    f"Duplicates dropped — Cloud: {res.get('n_duplicate_rows_dropped_cloud', 0):,} · "
                    f"On-Prem: {res.get('n_duplicate_rows_dropped_linc', 0):,}"
                )

                mdf = res.get("mismatch_df")
                if mdf is not None and not mdf.empty:
                    st.subheader("Mismatch detail (first 200 rows)")
                    st.dataframe(mdf.head(200), use_container_width=True, hide_index=True)
                elif res.get("n_failed", 0) == 0:
                    st.success("No cell-level mismatches in scope.")
                if res.get("mismatch_rows_truncated"):
                    st.warning(
                        "Mismatch table and CSV list at most 250,000 rows; total mismatch count in metrics is complete."
                    )


_QC_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qc_output")


def _save_outputs_to_qc_output(csv_exports: dict, pdf_bytes: bytes, pdf_filename: str) -> str:
    """Write all generated CSVs and the PDF to qc_output/. Returns the directory path."""
    os.makedirs(_QC_OUTPUT_DIR, exist_ok=True)
    for fname, content in (csv_exports or {}).items():
        with open(os.path.join(_QC_OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(content)
    with open(os.path.join(_QC_OUTPUT_DIR, pdf_filename), "wb") as f:
        f.write(pdf_bytes)
    return _QC_OUTPUT_DIR


@st_fragment
def _render_concordance_downloads(clin_res: dict) -> None:
    """Fragment-isolated concordance download buttons — clicking does not rerun the full app."""
    st.subheader("📥 Download concordance reports")
    st.caption("These reports cover Cloud vs On-Prem concordance only, "
               "separate from the general QC report.")

    conc_csvs = generate_concordance_csv_exports(clin_res)
    dl1, dl2, dl3, dl4, dl5 = st.columns(5)
    with dl1:
        st.download_button(
            "📄 Mismatches CSV",
            conc_csvs.get("concordance_mismatches.csv", b""),
            file_name="concordance_mismatches.csv",
            mime="text/csv",
            key="dl_conc_mismatches"
        )
    with dl2:
        st.download_button(
            "📄 Matches CSV",
            conc_csvs.get("concordance_matches.csv", b""),
            file_name="concordance_matches.csv",
            mime="text/csv",
            key="dl_conc_matches"
        )
    with dl3:
        st.download_button(
            "📄 Per-column CSV",
            conc_csvs.get("concordance_per_column_summary.csv", b""),
            file_name="concordance_per_column_summary.csv",
            mime="text/csv",
            key="dl_conc_col_summary"
        )
    with dl4:
        st.download_button(
            "📄 Summary CSV",
            conc_csvs.get("concordance_summary.csv", b""),
            file_name="concordance_summary.csv",
            mime="text/csv",
            key="dl_conc_summary"
        )
    with dl5:
        conc_pdf = generate_concordance_pdf_report(
            clin_res,
            label_cloud="Cloud",
            label_onprem="On-Prem"
        )
        st.download_button(
            "📋 Full PDF Report",
            conc_pdf,
            file_name=get_concordance_pdf_filename(),
            mime="application/pdf",
            key="dl_conc_pdf"
        )


@st_fragment
def _render_download_buttons(
    n_files,
    total_records,
    total_unique,
    common_count,
    qc_score,
    qc_status,
    header_result,
    record_metrics,
    overlap_data,
    mismatch_result,
    dup_analysis,
    results,
):
    st.subheader("Download reports")
    try:
        qc_summary = {
            "files_compared": n_files,
            "total_records": total_records,
            "unique_variants": total_unique,
            "common_in_all": common_count,
            "qc_score": qc_score,
            "qc_status": qc_status,
            "header_similarity_pct": header_result.get("header_similarity_pct", 0),
        }
        clin_state = results.get("clinical_concordance") or st.session_state.get("clinical_concordance")
        clin_mdf = None
        if clin_state and not clin_state.get("error") and clin_state.get("mismatch_df") is not None:
            m = clin_state["mismatch_df"]
            if hasattr(m, "empty") and not m.empty:
                clin_mdf = m
        csv_exports = generate_csv_exports(
            record_metrics,
            overlap_data or {},
            mismatch_result.get("mismatch_table"),
            dup_analysis or {},
            header_result.get("column_table"),
            overlap_data.get("unique_per_file", {}),
            overlap_data.get("missing_pairs", {}),
            qc_summary=qc_summary,
            clinical_mismatch_df=clin_mdf,
        )
        for fname, content in (csv_exports or {}).items():
            st.download_button(f"Download {fname}", content, file_name=fname, mime="text/csv", key=f"csv_{fname}")

        pdf_filename = get_pdf_filename()
        pdf_bytes = generate_pdf_report(
            qc_score=qc_score,
            qc_status=qc_status,
            n_files=n_files,
            total_records=total_records,
            total_unique=total_unique,
            common_count=common_count,
            record_metrics=record_metrics,
            header_similarity=header_result.get("header_similarity_pct", 0),
            header_table=header_result.get("column_table"),
            overlap_data=overlap_data or {},
            mismatch_count=mismatch_result.get("mismatch_count", 0),
            dup_analysis=dup_analysis or {},
            management_summary={
                "variant_consistency": "High" if qc_score >= 75 else "Medium" if qc_score >= 50 else "Low"
            },
            clinical_concordance=clin_state if clin_state else None,
        )
        st.download_button(
            "Download PDF report", pdf_bytes, file_name=pdf_filename, mime="application/pdf", key="pdf_dl"
        )

        # Save all outputs to qc_output/ directory
        try:
            out_dir = _save_outputs_to_qc_output(csv_exports, pdf_bytes, pdf_filename)
            st.caption(f"Outputs also saved to `{out_dir}/`")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Could not generate reports: {e}")


def _render_qc_results(results: dict) -> None:
    """Unpack QC results and render KPIs, tabs, and download buttons."""
    if not results or not isinstance(results, dict):
        st.error("No results to display.")
        return
    try:
        _render_qc_results_impl(results)
    except Exception as e:
        st.error(f"Could not display results: {e}")


def _render_qc_results_impl(results: dict) -> None:
    """Implementation of QC results rendering (called inside try/except)."""
    col_title, col_clear = st.columns([8, 1])
    with col_clear:
        if st.button("🔄 Clear & reload", key="clear_cache_btn"):
            for k in ["_qc_results", "_qc_file_signatures", "s3_resolved_paths", "clinical_concordance"]:
                st.session_state.pop(k, None)
            st.rerun()
    with col_title:
        pass

    record_metrics = results.get("record_metrics")
    unique_metrics = results.get("unique_metrics") or {}
    dup_analysis = results.get("dup_analysis") or {}
    header_result = results.get("header_result") or {}
    overlap_data = results.get("overlap_data") or {}
    mismatch_result = results.get("mismatch_result") or {}
    consistency_result = results.get("consistency_result") or {}
    missing_data = results.get("missing_data") or {}
    qc_score = results.get("qc_score", 0)
    qc_status = results.get("qc_status", "Unknown")
    total_records = results.get("total_records", 0)
    total_unique = results.get("total_unique", 0)
    common_count = results.get("common_count", 0)
    n_files = results.get("n_files", 0)
    dataframes = results.get("dataframes") or {}
    variant_keys = results.get("variant_keys") or {}

    total_dup_entries = 0
    if record_metrics is not None and not record_metrics.empty:
        try:
            tr = record_metrics.get("Total Rows")
            uv = record_metrics.get("Unique Variants")
            if tr is not None and uv is not None:
                total_dup_entries = int((tr - uv).sum())
        except (TypeError, ValueError, AttributeError):
            pass

    if header_result.get("has_mismatch"):
        st.markdown('<div class="warning-box">⚠️ Header mismatch detected across files. Check the Header QC tab.</div>', unsafe_allow_html=True)

    st.subheader("Key metrics")
    cols = st.columns(4)
    for i, (label, val) in enumerate([
        ("Total files", str(n_files)),
        ("Total records", f"{total_records:,}"),
        ("Unique variants (union)", f"{total_unique:,}"),
        ("Duplicate variant entries", f"{total_dup_entries:,}"),
    ]):
        with cols[i]:
            st.markdown(f'<div class="kpi-card"><h3>{label}</h3><div class="value">{val}</div></div>', unsafe_allow_html=True)
    cols2 = st.columns(4)
    unique_per_file = unique_metrics.get("per_file", {})
    unique_in_one = sum(unique_per_file.get(n, {}).get("only_in_this_file", 0) for n in unique_metrics.get("names", []))
    for i, (label, val) in enumerate([
        ("Variants common in all files", f"{common_count:,}"),
        ("Variants unique to one file", f"{unique_in_one:,}"),
        ("Header consistency score", f"{header_result.get('header_similarity_pct', 0)}%"),
        ("QC Score", f"{qc_score} — {qc_status}"),
    ]):
        with cols2[i]:
            st.markdown(f'<div class="kpi-card"><h3>{label}</h3><div class="value">{val}</div></div>', unsafe_allow_html=True)

    with st.expander("📋 Management summary", expanded=True):
        consistency_label = "High" if qc_score >= 75 else "Medium" if qc_score >= 50 else "Low"
        st.markdown(f"""
        - **Files compared:** {n_files}
        - **Total variants analyzed:** {total_unique:,}
        - **Variants common across all files:** {common_count:,}
        - **Variant consistency:** {consistency_label}
        - **QC Score:** {qc_score}/100 ({qc_status})
        """)

    # CONCORDANCE — auto-run clinical concordance display
    clin_result = results.get("clinical_concordance")
    if clin_result and not clin_result.get("error") and clin_result.get("total_tests"):
        st.session_state["clinical_concordance"] = clin_result
        with st.expander("Executive validation — Cloud vs On-Prem (all genes, automatic)", expanded=True):
            res = clin_result

            # CONCORDANCE — Row 1: top-level KPIs
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Overall Concordance", f"{res.get('concordance_pct', 0):.4f}%")
            with m2:
                st.metric("Perfectly Matched Rows",
                          f"{res.get('n_perfect_match_rows', 0):,}",
                          help="Rows where every compared column matched exactly")
            with m3:
                st.metric("Rows with Mismatches",
                          f"{res.get('n_mismatch_rows', 0):,}",
                          help="Rows where at least one column differed")

            m4, m5, m6 = st.columns(3)
            with m4:
                st.metric("Strict Field Mismatches",
                          res.get('n_failed_strict', 0),
                          help="Clinical fields (CDNA_CHG, AA_CHG, HGVS) — zero tolerance")
            with m5:
                st.metric("Cell-level Tests", f"{res.get('total_tests', 0):,}")
            with m6:
                st.metric("Matched Rows (inner join)", f"{res.get('n_rows_matched', 0):,}")

            # CONCORDANCE — Row 2: validation status banner
            n_strict = res.get('n_failed_strict', 0)
            concordance = res.get('concordance_pct', 0)
            if n_strict == 0 and concordance >= 99.0:
                st.markdown(
                    '<div class="success-box">✅ <b>PASS</b> — No strict clinical field '
                    'mismatches. Concordance ≥ 99%. Files are validated for upload.</div>',
                    unsafe_allow_html=True
                )
            elif n_strict == 0 and concordance >= 95.0:
                st.markdown(
                    '<div class="warning-box">⚠️ <b>REVIEW</b> — No strict mismatches but '
                    f'concordance is {concordance:.2f}%. Investigate non-strict mismatches '
                    'before upload.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="warning-box">❌ <b>FAIL</b> — Strict clinical field '
                    f'mismatches detected ({n_strict}). Do NOT upload until resolved.</div>',
                    unsafe_allow_html=True
                )

            # CONCORDANCE — Row 3: per-column concordance table
            col_pcts = res.get('col_concordance_pct', {})
            col_counts = res.get('col_mismatch_counts', {})
            if col_pcts:
                col_df = pd.DataFrame([
                    {
                        "Column": col,
                        "Concordance %": f"{pct:.2f}",
                        "Mismatches": col_counts.get(col, 0),
                        "Status": "✅" if pct == 100.0 else ("⚠️" if pct >= 95.0 else "❌")
                    }
                    for col, pct in sorted(col_pcts.items(), key=lambda x: x[1])
                ])
                st.subheader("Per-column concordance")
                st.caption("Sorted by concordance ascending — worst columns first.")
                st.dataframe(col_df, use_container_width=True, hide_index=True)

            # CONCORDANCE — Row 4: strict field failures
            strict_vars = res.get('strict_fail_variants', [])
            if strict_vars:
                st.subheader("⚠️ Variants with strict field failures")
                st.caption("These variants have mismatches in CDNA_CHG / AA_CHG / HGVS "
                           "— clinical report fields. Must be resolved before upload.")
                st.dataframe(pd.DataFrame({"Variant Key": strict_vars}),
                             use_container_width=True, hide_index=True)

            # CONCORDANCE — Row 5: mismatch and match detail tables
            tab_mm, tab_match = st.tabs(["❌ Mismatches", "✅ Matches"])
            with tab_mm:
                mdf = res.get("mismatch_df")
                if mdf is not None and not mdf.empty:
                    st.caption(f"Showing first 200 of {res.get('n_mismatch_rows', 0):,} "
                               f"mismatch rows. Download full CSV below.")
                    st.dataframe(mdf.head(200), use_container_width=True, hide_index=True)
                else:
                    st.success("No mismatches found.")
            with tab_match:
                match_df = res.get("match_df")
                if match_df is not None and not match_df.empty:
                    st.caption(f"{res.get('n_perfect_match_rows', 0):,} rows matched "
                               f"perfectly across all {len(res.get('col_concordance_pct', {}))} "
                               f"compared columns.")
                    st.dataframe(match_df.head(200), use_container_width=True, hide_index=True)

            # CONCORDANCE — Row 6: excluded columns note
            excl = res.get('excluded_cols', [])
            if excl:
                st.caption(f"ℹ️ Excluded from comparison: {', '.join(excl)}")

            for w in res.get("warnings") or []:
                st.warning(w)

            # CONCORDANCE — download buttons (fragment-isolated to prevent full rerun)
            _render_concordance_downloads(res)
    elif clin_result and clin_result.get("error"):
        with st.expander("Executive validation — Cloud vs On-Prem", expanded=False):
            st.error(f"Clinical concordance: {clin_result['error']}")
    else:
        with st.expander("Executive validation — Cloud vs On-Prem", expanded=False):
            st.info("Clinical concordance was not run (requires 2 files with matching columns).")

    tab_overview, tab_variant, tab_dup, tab_header, tab_mismatch, tab_missing, tab_reports = st.tabs([
        "Overview", "Variant comparison", "Duplicate analysis", "Header QC",
        "Mismatch analysis", "Missing data", "QC reports",
    ])

    with tab_overview:
        st.subheader("Record-level summary")
        if record_metrics is not None and not record_metrics.empty:
            st.dataframe(record_metrics, use_container_width=True, hide_index=True)
        st.subheader("Unique variant summary")
        per_file = unique_metrics.get("per_file")
        if per_file and isinstance(per_file, dict):
            rows = []
            for name, d in per_file.items():
                if isinstance(d, dict):
                    rows.append({
                        "File": str(name),
                        "Unique variants": d.get("unique_variants", 0),
                        "Only in this file": d.get("only_in_this_file", 0),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("QC score")
        score_class = "qc-score-excellent" if qc_score >= 90 else "qc-score-good" if qc_score >= 75 else "qc-score-fair" if qc_score >= 50 else "qc-score-poor"
        st.markdown(f'<p class="{score_class}">QC Score: {qc_score}/100 — {qc_status}</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=qc_score, number={"suffix": "/100"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2c5282"}, "steps": [{"range": [0, 50], "color": "#fecaca"}, {"range": [50, 75], "color": "#fef08a"}, {"range": [75, 90], "color": "#bbf7d0"}, {"range": [90, 100], "color": "#86efac"}]}))
        fig.update_layout(height=250, margin=dict(l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab_variant:
        st.subheader("Variant overlap")
        names = overlap_data.get("names", [])
        sets = overlap_data.get("sets", {})
        if len(names) >= 2:
            matrix_data = [[len(sets.get(a, set()) & sets.get(b, set())) if a != b else len(sets.get(a, set())) for b in names] for a in names]
            fig = px.imshow(matrix_data, x=names, y=names, labels=dict(x="", y=""), text_auto="d", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Common variants (in all files)")
        st.metric("Count", f"{overlap_data.get('common_count', 0):,}")
        common_ids = list(overlap_data.get("common_in_all", set()))[:500]
        if common_ids:
            st.dataframe(pd.DataFrame({"variant_id": common_ids}), use_container_width=True, hide_index=True)
        st.subheader("Unique per file")
        for name, ids in overlap_data.get("unique_per_file", {}).items():
            with st.expander(f"{name} — {len(ids):,} unique"):
                st.dataframe(pd.DataFrame({"variant_id": list(ids)[:500]}), hide_index=True)

    with tab_dup:
        st.subheader("Duplicate variant analysis")
        for name, d in dup_analysis.items():
            st.markdown(f"**{name}**")
            st.write(f"Variants with duplicate rows: {d.get('n_duplicated_variants', 0):,}")
            st.write(f"Total duplicate entries: {d.get('total_duplicate_entries', 0):,}")
            top = d.get("top_duplicated")
            if top is not None and len(top) > 0:
                st.dataframe(pd.DataFrame({"variant_key": top.index.tolist(), "count": top.values}), hide_index=True)
            hist = d.get("histogram_counts")
            if hist is not None and len(hist) > 0:
                fig = px.bar(x=hist.index.astype(str), y=hist.values, labels={"x": "Duplicate count", "y": "Number of variants"})
                st.plotly_chart(fig, use_container_width=True)
            st.divider()

    with tab_header:
        st.subheader("Header comparison")
        st.metric("Header similarity", f"{header_result.get('header_similarity_pct', 0)}%")
        col_table = header_result.get("column_table")
        if col_table is not None and not (hasattr(col_table, "empty") and col_table.empty):
            st.dataframe(col_table, use_container_width=True, hide_index=True)

    with tab_mismatch:
        st.subheader("Column value mismatches")
        mdf = mismatch_result.get("mismatch_table")
        if mdf is not None and not mdf.empty:
            st.dataframe(mdf.head(500), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 500 of {len(mdf):,} mismatches.")
        else:
            st.info("No mismatches detected.")
        st.subheader("Variant consistency (grouped)")
        cdf = consistency_result.get("inconsistency_df")
        if cdf is not None and not cdf.empty:
            st.dataframe(cdf.head(300), use_container_width=True, hide_index=True)
        else:
            st.info("No annotation inconsistencies detected.")

    with tab_missing:
        st.subheader("Missing data")
        na_df = missing_data.get("na_pct_df")
        if na_df is not None and not na_df.empty:
            fig = px.imshow(na_df.T, labels=dict(x="Column", y="File", color="NA %"), color_continuous_scale="Reds", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        st.write("Empty columns (100% NA in all files):", missing_data.get("empty_columns", [])[:30])
        st.write("High NA columns (≥50% in any file):", missing_data.get("high_na_columns", [])[:30])

    with tab_reports:
        st.subheader("Data preview")
        file_list = list(dataframes.keys()) if dataframes else []
        file_sel = st.selectbox("Select file", file_list, key="preview_file") if file_list else None
        if file_sel and file_sel in dataframes:
            st.dataframe(dataframes[file_sel].head(100), use_container_width=True, hide_index=True)
        st.subheader("Search variant")
        search_key = st.text_input("Variant key (CHROM_START_REF_ALT)", key="search_key")
        if search_key and variant_keys:
            for name, keys in variant_keys.items():
                try:
                    if hasattr(keys, "values") and search_key in keys.values:
                        st.write(f"**{name}**: found")
                        idx = keys[keys == search_key].index
                        if name in dataframes:
                            st.dataframe(dataframes[name].loc[idx], use_container_width=True, hide_index=True)
                except Exception:
                    pass
        st.subheader("Filter by gene")
        gene_col = None
        for df in dataframes.values():
            for c in ["GENE_NAME", "GENE", "Gene"]:
                if c in df.columns:
                    gene_col = c
                    break
            if gene_col:
                break
        if gene_col and dataframes:
            try:
                all_genes = set()
                for df in dataframes.values():
                    if gene_col in getattr(df, "columns", []):
                        all_genes.update(df[gene_col].dropna().astype(str).unique().tolist())
                genes_sorted = sorted(all_genes)[:500]
                gene_sel = st.selectbox("Gene", genes_sorted, key="gene_sel") if genes_sorted else None
                if gene_sel:
                    for name, df in dataframes.items():
                        if gene_col in getattr(df, "columns", []):
                            sub = df[df[gene_col].astype(str) == gene_sel]
                            if len(sub) > 0:
                                st.write(f"**{name}**: {len(sub)} rows")
                                st.dataframe(sub.head(50), hide_index=True)
            except Exception:
                pass
        _render_download_buttons(
            n_files,
            total_records,
            total_unique,
            common_count,
            qc_score,
            qc_status,
            header_result,
            record_metrics,
            overlap_data,
            mismatch_result,
            dup_analysis,
            results,
        )

    if dup_analysis:
        dup_counts = [d.get("n_duplicated_variants", 0) for d in dup_analysis.values()]
        if dup_counts and len(set(dup_counts)) > 1 and max(dup_counts) - min(dup_counts) > 100:
            st.warning("Possible pipeline version difference: duplication patterns differ strongly across files.")


@st.cache_data(ttl=3600)
def run_full_qc(file_signatures, _dataframes_list):
    """
    Run all QC analyses. Cache key is file_signatures only (paths+mtimes+sizes or upload name+size);
    DataFrames are passed via _dataframes_list so Streamlit does not hash full frames.
    """
    if not _dataframes_list or len(_dataframes_list) < 2:
        return None
    # file_signatures is hashed by @st.cache_data for invalidation only (not read here).
    try:
        dataframes = {name: df for name, df in _dataframes_list}
    except Exception as e:
        return {"error": str(e)}
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

        import gc

        mismatch_result = column_mismatch_detection(dataframes, variant_keys, common_ids)
        consistency_result = variant_consistency_analysis(dataframes, variant_keys, common_ids)
        del common_ids
        gc.collect()

        missing_data = missing_data_analysis(dataframes)
    except Exception as e:
        return {"error": str(e)}

    # Auto-run clinical concordance (all genes) on the first two files
    clinical_concordance_result = None
    names = list(dataframes.keys())
    if len(names) >= 2:
        try:
            from clinical_concordance import (
                detect_gene_column,
                run_pairwise_concordance,
            )

            df_cloud = dataframes[names[0]]
            df_linc = dataframes[names[1]]
            gene_col = detect_gene_column(df_cloud)
            gene_col_l = detect_gene_column(df_linc)
            if gene_col and gene_col == gene_col_l:
                pass  # same column on both sides
            else:
                gene_col = None  # no gene column needed for compare_all_genes=True

            # CONCORDANCE — auto-run with explicit params
            clinical_concordance_result = run_pairwise_concordance(
                df_cloud,
                df_linc,
                compare_all_genes=True,
                label_cloud="Cloud",
                label_linc="On-Prem",
                strict_columns=None,
                compare_columns=None,
                tolerance_pct=10.0,
                restrict_ontarget=True,
            )
        except Exception as e:
            clinical_concordance_result = {"error": str(e), "mismatch_df": pd.DataFrame()}

    total_records = sum(len(df) for df in dataframes.values())
    total_unique = unique_metrics.get("total_unique_across_all", 0)
    common_count = unique_metrics.get("common_in_all", 0)
    dup_penalty = 0.0
    if dup_analysis:
        dup_counts = [d.get("n_duplicated_variants", 0) for d in dup_analysis.values()]
        if dup_counts and len(set(dup_counts)) > 1 and max(dup_counts) - min(dup_counts) > 100:
            dup_penalty = 10.0  # pipeline drift
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
        "clinical_concordance": clinical_concordance_result,
    }


def main():
    # Professional header strip
    st.markdown("""
    <div class="main-header">
        <h1>🧬 VariMAT QC Dashboard</h1>
        <p>Load 2–10 VariMAT files (.txt, .tsv, .txt.gz) via <b>S3</b>, <b>upload / browse</b>, or <b>local server paths</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Quick start")
        st.markdown("Pick **one** load method: **S3**, **upload / browse**, or **local paths** on the server. Columns: **CHROM**, **START**, **REF**, **ALT**.")
        st.markdown("---")

    local_paths = []

    from s3_loader import download_s3_file, get_s3_config, list_s3_files, list_varimat_folders

    s3_cfg = get_s3_config()

    _METHOD_S3 = "s3"
    _METHOD_UPLOAD = "upload"
    _METHOD_PATHS = "paths"
    _method_labels = {
        _METHOD_S3: "☁️ S3 (bucket)",
        _METHOD_UPLOAD: "📤 Local file upload & browse",
        _METHOD_PATHS: "📁 Local paths (server disk)",
    }
    _method_options = [_METHOD_S3, _METHOD_UPLOAD, _METHOD_PATHS]

    # Default to upload method if no previous selection stored
    if "file_load_method" not in st.session_state:
        st.session_state["file_load_method"] = _METHOD_UPLOAD

    load_method = st.radio(
        "Choose how to load files",
        options=_method_options,
        format_func=lambda k: _method_labels[k],
        horizontal=True,
        key="file_load_method",
    )

    # When method changes, clear all cross-method session state to avoid
    # stale state from previous method bleeding into the new one
    _prev_method = st.session_state.get("_prev_load_method")
    if _prev_method and _prev_method != load_method:
        for _k in [
            "s3_resolved_paths",
            "s3_folder_list",
            "s3_file_list",
            "s3_last_prefix",
            "s3_selected_keys",
            "clinical_concordance",
            "_qc_results",
            "_qc_file_signatures",
        ]:
            st.session_state.pop(_k, None)
    st.session_state["_prev_load_method"] = load_method

    path_input = ""
    use_paths = False
    uploaded = None

    if load_method == _METHOD_S3:
        if not s3_cfg:
            st.warning(
                "**S3 is not configured.** Set **`S3_BUCKET`** and AWS credentials as environment variables "
                "or in **`.streamlit/secrets.toml`**. See **`.env.example`** in this repo. "
                "Restart Streamlit after changing them."
            )
        else:
            # MIXED-SOURCE: combined S3 + browser upload in one expander
            with st.expander("☁️ Load files — S3 + optional browser upload", expanded=True):
                # MIXED-SOURCE — Uploader first at top of expander so widget state is stable (not after long S3 folder branches)
                st.markdown("#### Step 1 — Optional: upload files from your computer")
                st.caption("Use this to compare an S3 file against a local file not yet in S3.")
                _n_s3_prev = len(st.session_state.get("s3_selected_keys") or [])
                _max_upload_prev = max(1, 10 - _n_s3_prev)
                uploaded_mixed = st.file_uploader(
                    f"Upload up to {_max_upload_prev} additional file(s)",
                    type=["txt", "tsv", "gz"],
                    accept_multiple_files=True,
                    key="s3_mixed_uploader",
                    help="These are combined with the S3 files you select in Step 2.",
                )

                st.divider()

                # MIXED-SOURCE — Section A: S3 folder + file picker (0–9 S3 objects)
                st.markdown("#### Step 2 — Pick files from S3 (optional)")
                base_prefix = s3_cfg["prefix"] or ""
                st.caption(
                    f"Bucket: `{s3_cfg['bucket']}`  Prefix: `{base_prefix or '/'}` — "
                    "skip this step if you already chose 2+ files in Step 1."
                )

                _, col_btn = st.columns([6, 1])
                with col_btn:
                    if st.button("🔄 Refresh", key="s3_refresh"):
                        for k in ("s3_folder_list", "s3_file_list", "s3_last_prefix", "s3_files_for_prefix"):
                            st.session_state.pop(k, None)
                        st.rerun()

                if "s3_folder_list" not in st.session_state:
                    with st.spinner("Scanning S3 folders (up to 5 levels)…"):
                        st.session_state["s3_folder_list"] = list_varimat_folders(
                            s3_cfg["bucket"], base_prefix, max_depth=5
                        )

                err = st.session_state.pop("s3_prefix_error", None)
                if err:
                    st.error(f"S3 folder scan failed: {err}")

                folders = st.session_state.get("s3_folder_list", [])
                chosen_display = base_prefix or "/"

                if not folders:
                    st.info("No folders found — listing files directly under prefix.")
                    selected_prefix = base_prefix
                else:
                    folder_display_names = [
                        (f.removeprefix(base_prefix) if base_prefix else f).rstrip("/") or f.rstrip("/")
                        for f in folders
                    ]
                    chosen_display = st.selectbox(
                        "Run folder",
                        folder_display_names,
                        key="s3_folder_sel",
                        help="Scanned up to 5 levels deep; shows folders named 'varimat' first.",
                    )
                    selected_prefix = folders[folder_display_names.index(chosen_display)]

                if st.session_state.get("s3_last_prefix") != selected_prefix:
                    st.session_state.pop("s3_file_list", None)
                    st.session_state["s3_last_prefix"] = selected_prefix

                if "s3_file_list" not in st.session_state:
                    with st.spinner(f"Listing files in `{chosen_display}`…"):
                        st.session_state["s3_file_list"] = list_s3_files(
                            s3_cfg["bucket"], selected_prefix
                        )

                files_for_select = st.session_state.get("s3_file_list", [])

                if not folders and not files_for_select:
                    st.warning("No files or folders found under the configured prefix.")
                elif not files_for_select:
                    st.info(
                        "No .txt / .tsv / .gz files in this folder — use **Step 1** uploads only, "
                        "or pick another run folder."
                    )
                else:
                    file_df = pd.DataFrame(files_for_select)[["key", "size_mb", "last_modified"]]
                    file_df.columns = ["S3 Key", "Size (MB)", "Last Modified"]
                    st.dataframe(file_df, use_container_width=True, hide_index=True)

                # Always render multiselect so widget key stays registered (upload-only runs still work).
                s3_options = [f["key"] for f in files_for_select]
                _s3_opts_display = (
                    s3_options
                    if s3_options
                    else ["(No .txt / .tsv / .gz here — use Step 1 uploads or another folder)"]
                )
                _raw_s3_sel = st.multiselect(
                    "Select files from S3 (optional — 0 needed if Step 1 has your files)",
                    options=_s3_opts_display,
                    max_selections=9 if s3_options else 1,
                    key="s3_selected_keys",
                    disabled=not bool(s3_options),
                    help="Mix with Step 1: e.g. 2 uploads, or 1 S3 + 1 upload, or 2 from S3.",
                )
                s3_selected_keys = [k for k in (_raw_s3_sel or []) if k in s3_options][:9]

                max_upload = max(1, 10 - (len(s3_selected_keys) if s3_selected_keys else 0))
                um_list = list(uploaded_mixed or [])
                if len(um_list) > max_upload:
                    st.warning(
                        f"Max {max_upload} uploads allowed when "
                        f"{len(s3_selected_keys) if s3_selected_keys else 0} S3 file(s) selected. "
                        f"Only first {max_upload} used."
                    )
                    um_list = um_list[:max_upload]

                n_s3_btn = len(s3_selected_keys) if s3_selected_keys else 0
                n_up_btn = len(um_list)
                total_btn = n_s3_btn + n_up_btn
                if total_btn < 2:
                    st.info(
                        f"👆 Select at least 2 files total. Currently: {total_btn} selected "
                        f"({n_s3_btn} from S3, {n_up_btn} uploaded)."
                    )

                st.divider()

                # MIXED-SOURCE — Section C: Load & Compare → /tmp paths + existing local pipeline
                st.markdown("#### Step 3 — Load & Compare")

                if s3_selected_keys:
                    st.markdown("**From S3:**")
                    for k in s3_selected_keys:
                        fname = k.split("/")[-1]
                        size_mb = next(
                            (float(f["size_mb"]) for f in files_for_select if f["key"] == k),
                            0.0,
                        )
                        st.caption(f"  ☁️ {fname}  ({size_mb:.0f} MB)")
                if um_list:
                    st.markdown("**From browser:**")
                    for f in um_list:
                        # Do not call getvalue() here — it can disturb the buffer before save uses f.read().
                        nbytes = getattr(f, "size", None) or 0
                        size_mb = (nbytes / (1024 * 1024)) if nbytes else 0.0
                        st.caption(
                            f"  💻 {f.name}  ({size_mb:.0f} MB)"
                            if nbytes
                            else f"  💻 {f.name}"
                        )

                # Re-evaluate counts immediately before the button (um_list matches what we will save/load).
                n_s3_btn = len(s3_selected_keys) if s3_selected_keys else 0
                n_up_btn = len(um_list)
                total_btn = n_s3_btn + n_up_btn

                load_btn = st.button(
                    "🚀 Load & Compare",
                    type="primary",
                    key="s3_mixed_load_btn",
                    disabled=total_btn < 2,
                )

                # MIXED-SOURCE — persist combined paths; uploads written as real files for parquet sidecars
                if load_btn and total_btn >= 2:
                    combined_local_paths: list[str] = []
                    load_errors: list[str] = []
                    tmp_root = os.environ.get("TMPDIR", "/tmp")

                    if s3_selected_keys:
                        dl_progress = st.progress(0)
                        dl_status = st.empty()
                        n_s3 = len(s3_selected_keys)
                        for i, key in enumerate(s3_selected_keys):
                            fname = key.split("/")[-1]
                            dl_status.text(f"Downloading from S3: {fname} ({i + 1}/{n_s3})…")
                            file_bar = st.progress(0)
                            try:
                                local_path = download_s3_file(
                                    s3_cfg["bucket"], key, file_progress=file_bar
                                )
                                combined_local_paths.append(local_path)
                            except Exception as e:
                                load_errors.append(f"S3 {fname}: {e}")
                            dl_progress.progress((i + 1) / n_s3)
                        try:
                            dl_progress.empty()
                        except Exception:
                            pass
                        try:
                            dl_status.empty()
                        except Exception:
                            pass

                    if um_list:
                        up_status = st.empty()
                        for f in um_list:
                            up_status.text(f"Saving uploaded file: {f.name}…")
                            tmp_path = os.path.join(tmp_root, f.name)
                            try:
                                f.seek(0)
                                raw = f.read()
                                if not raw:
                                    load_errors.append(f"Upload {f.name}: file is empty")
                                    continue
                                with open(tmp_path, "wb") as out:
                                    out.write(raw)
                                if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
                                    load_errors.append(f"Upload {f.name}: failed to write to /tmp")
                                    continue
                                combined_local_paths.append(tmp_path)
                            except Exception as e:
                                load_errors.append(f"Upload {f.name}: {e}")
                        try:
                            up_status.empty()
                        except Exception:
                            pass

                    for msg in load_errors:
                        st.error(msg)

                    if len(combined_local_paths) >= 2:
                        st.session_state["s3_resolved_paths"] = combined_local_paths
                        st.rerun()
                    else:
                        st.error("Not enough files loaded successfully (need at least 2).")

    elif load_method == _METHOD_UPLOAD:
        st.caption("Upload from your computer (browser). For large files on the same machine as the server, use **Local paths** instead.")
        st.warning(
            "**Large files (hundreds of MB or more):** browser upload often fails on hosted apps with **502 / network errors** "
            "because the platform proxy times out before the upload finishes. "
            "Use **☁️ S3** or **📁 Local paths** on the server instead — they do not send the whole file through the browser."
        )
        uploaded = st.file_uploader(
            "Browse or drag & drop VariMAT files",
            type=["txt", "tsv", "gz"],
            accept_multiple_files=True,
            help="Select 2–10 files (.txt, .tsv, .gz). Max size is set in Streamlit server config.",
        )

    else:
        st.caption("Paths must exist on the machine running Streamlit (e.g. laptop or Railway volume). No browser upload.")
        path_input = st.text_area(
            "Enter one absolute file path per line",
            height=100,
            placeholder="/path/to/file1.txt\n/path/to/file2.txt",
            help="Server reads these paths directly from disk.",
        )
        use_paths = st.button("Load from paths", key="load_paths")

    if "s3_resolved_paths" in st.session_state:
        local_paths = st.session_state.pop("s3_resolved_paths")

    if (path_input or "").strip() and use_paths:
        try:
            lines = (path_input or "").strip().splitlines()
            local_paths = [p.strip() for p in lines if p and p.strip()]
            local_paths = [os.path.expanduser(p) for p in local_paths if p][:10]
            # Validate: warn about non-existent paths before loading
            missing = [p for p in local_paths if not os.path.isfile(p)]
            if missing:
                for m in missing[:5]:
                    st.warning(f"Path not found or not a file: {m}")
                if len(missing) > 5:
                    st.warning(f"… and {len(missing) - 5} more path(s) not found.")
                local_paths = [p for p in local_paths if os.path.isfile(p)]
        except Exception as e:
            st.error(f"Invalid paths: {e}")
            local_paths = []

    if local_paths:
        # Load from local paths (no upload — server reads from disk)
        if len(local_paths) < 2:
            st.warning("Enter at least 2 paths for comparison.")
        else:
            # Check if these exact files were already parsed this session (canonical path order)
            try:
                _incoming_sig = _qc_local_paths_signature(local_paths)
            except Exception:
                _incoming_sig = None

            if (
                _incoming_sig
                and st.session_state.get("_qc_file_signatures") == _incoming_sig
                and st.session_state.get("_qc_results") is not None
            ):
                st.success("✅ Using cached results (files unchanged).")
                results_qc = st.session_state["_qc_results"]
                if results_qc is None:
                    st.stop()
                if results_qc.get("error"):
                    st.error(results_qc["error"])
                else:
                    _render_qc_results(results_qc)
                st.stop()

            # else: fall through to normal parse + QC pipeline below
            file_info = []
            dataframes = {}
            errors = []
            seen_names = {}
            ordered_success = []
            max_workers = 1  # sequential on Railway — prevents double memory peak
            progress_bar = st.progress(0)
            st.caption("Loading from disk (thread pool; Polars releases GIL during I/O)…")
            with st.spinner("Reading files from disk (no upload)…"):
                # Thread pool: avoids process spawn OOM on small Railway containers; cap workers for memory.
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(load_varimat_path_worker, (i, p)) for i, p in enumerate(local_paths)]
                    results = []
                    n_paths = len(local_paths)
                    for k, future in enumerate(as_completed(futures), start=1):
                        try:
                            results.append(future.result())
                        except Exception as e:
                            results.append((9999, "", "unknown", None, str(e), 0.0))
                        progress_bar.progress(min(k / n_paths, 1.0))
                results.sort(key=lambda x: int(x[0]) if x[0] is not None else 9999)
                for _ord, path, name, df, err, size_mb in results:
                    display_name = name or "unknown"
                    if name in seen_names:
                        seen_names[name] += 1
                        display_name = f"{name} ({seen_names[name]})"
                    else:
                        seen_names[name] = 1
                    if err or df is None:
                        errors.append((display_name, err or "Load failed"))
                        file_info.append({"File": display_name, "Records": "—", "Status": "❌ Error", "Size (MB)": f"{size_mb:.2f}"})
                    else:
                        dataframes[display_name] = df
                        if path and os.path.isfile(path):
                            ordered_success.append((os.path.abspath(path), display_name, df))
                        file_info.append({
                            "File": display_name,
                            "Records": f"{len(df):,}",
                            "Status": "✅ OK",
                            "Size (MB)": f"{size_mb:.2f}",
                        })

            st.subheader("Upload status (loaded from local paths)")
            if file_info:
                st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
            for name, msg in errors:
                st.error(f"**{name}**: {msg}")

            if len(dataframes) == 0 and errors:
                st.error("No files could be loaded. Check paths and file format (CHROM, START, REF, ALT columns required).")
                st.stop()

            if len(dataframes) >= 2:
                try:
                    if len(ordered_success) < 2:
                        st.error("Need at least 2 valid files to run QC.")
                    else:
                        # Store parsed dataframes in session_state so button clicks
                        # don't re-trigger file parsing (which causes OOM on Railway)
                        results_qc = None
                        _sig = _qc_local_paths_signature([p for p, _, _ in ordered_success])
                        if _sig is None:
                            st.error("Could not build file signatures for QC cache.")
                        elif st.session_state.get("_qc_file_signatures") != _sig:
                            file_signatures = _sig
                            dataframes_list = tuple((dn, d) for _, dn, d in ordered_success)
                            results_qc = run_full_qc(file_signatures, _dataframes_list=dataframes_list)
                            st.session_state["_qc_results"] = results_qc
                            st.session_state["_qc_file_signatures"] = _sig
                        else:
                            results_qc = st.session_state.get("_qc_results")

                        if _sig is not None:
                            if results_qc is None:
                                st.error("QC could not run (e.g. missing required columns).")
                            elif results_qc.get("error"):
                                st.error(results_qc["error"])
                            else:
                                _render_qc_results(results_qc)
                except Exception as e:
                    st.error(f"QC step failed: {e}")
            else:
                st.stop()
        st.stop()

    if load_method == _METHOD_UPLOAD:
        if not uploaded:
            st.info("👆 Use **Browse files** or drag & drop at least 2 VariMAT files.")
            return

        if len(uploaded) < 2:
            st.warning("Please upload at least 2 files for comparison.")
            return
        if len(uploaded) > 10:
            st.warning("Maximum 10 files allowed. Only the first 10 will be used.")
            uploaded = uploaded[:10]

        # Write uploads to /tmp one at a time (sequential) to avoid
        # holding multiple decompressed files in RAM simultaneously.
        # Then hand off to the local path pipeline (parquet cache + polars).
        tmp_root = os.environ.get("TMPDIR", "/tmp")
        tmp_paths = []
        write_errors = []

        write_bar = st.progress(0)
        write_status = st.empty()
        n_files = len(uploaded)

        for i, f in enumerate(uploaded):
            write_status.text(f"Saving {f.name} to server ({i + 1}/{n_files})…")
            tmp_path = os.path.join(tmp_root, f.name)
            try:
                f.seek(0)
                # Stream write in 64 MB chunks — never holds full file in RAM
                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = f.read(64 * 1024 * 1024)  # 64 MB chunks
                        if not chunk:
                            break
                        out.write(chunk)
                if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
                    write_errors.append(f"{f.name}: failed to write to /tmp")
                else:
                    tmp_paths.append(tmp_path)
            except Exception as e:
                write_errors.append(f"{f.name}: {e}")
            write_bar.progress((i + 1) / n_files)

        try:
            write_bar.empty()
            write_status.empty()
        except Exception:
            pass

        for msg in write_errors:
            st.error(msg)

        if len(tmp_paths) < 2:
            st.error("Not enough files saved successfully (need at least 2).")
            return

        # Hand off to local path pipeline — identical to S3 download path
        st.session_state["s3_resolved_paths"] = tmp_paths
        st.rerun()

    elif load_method == _METHOD_PATHS:
        st.info("Enter **local server paths** (one per line) and click **Load from paths**.")
        return
    elif load_method == _METHOD_S3:
        if "s3_resolved_paths" not in st.session_state:
            if s3_cfg:
                st.info("Select files from the S3 list and click **Load selected files from S3**.")
            return
    else:
        return


if __name__ == "__main__":
    main()
