"""
VariMAT QC Tool - Generate QC reports in CSV and PDF (executive-ready layout).
"""

import io
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# Modern executive palette
_CLR_HEADER = colors.HexColor("#1e3a5f")
_CLR_HEADER_TEXT = colors.HexColor("#f8fafc")
_CLR_ROW_ALT = colors.HexColor("#f1f5f9")
_CLR_GRID = colors.HexColor("#cbd5e1")


def _df_to_table_data(df: pd.DataFrame, max_rows: int = 50) -> list:
    """Convert DataFrame to list of lists for ReportLab Table."""
    if df is None or df.empty:
        return [["No data"]]
    df = df.head(max_rows)
    return [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()


def _table_style_modern() -> list:
    return [
        ("BACKGROUND", (0, 0), (-1, 0), _CLR_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), _CLR_HEADER_TEXT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _CLR_ROW_ALT]),
        ("GRID", (0, 0), (-1, -1), 0.5, _CLR_GRID),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]


def _safe_table(data: list, col_widths: list, font_size: int = 7,
                rows_per_chunk: int = 45) -> list:
    """Build one or more Tables from *data* so no single table exceeds
    *rows_per_chunk* body rows.  Headers repeat on every chunk.
    Returns a list of flowables (Table + Spacer pairs).
    """
    if not data or len(data) < 2:
        return []
    header = data[0]
    body_rows = data[1:]
    flowables: list = []
    style_base = _table_style_modern() + [("FONTSIZE", (0, 0), (-1, -1), font_size)]
    for start in range(0, len(body_rows), rows_per_chunk):
        chunk = [header] + body_rows[start : start + rows_per_chunk]
        t = Table(chunk, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle(style_base))
        flowables.append(t)
        if start + rows_per_chunk < len(body_rows):
            flowables.append(Spacer(1, 0.08 * inch))
    return flowables


def generate_csv_exports(
    record_metrics: pd.DataFrame,
    overlap_data: dict,
    mismatch_df: pd.DataFrame,
    duplicate_summary: dict,
    header_table: pd.DataFrame,
    unique_per_file: dict,
    missing_pairs: dict,
    qc_summary: dict = None,
    clinical_mismatch_df: Optional[pd.DataFrame] = None,
) -> dict[str, str]:
    """
    Generate CSV content strings for download. Returns dict of filename -> csv_string.
    """
    out = {}
    if qc_summary:
        qc_df = pd.DataFrame([qc_summary])
        out["qc_summary.csv"] = qc_df.to_csv(index=False)
    if record_metrics is not None and not record_metrics.empty:
        out["record_metrics.csv"] = record_metrics.to_csv(index=False)
    if overlap_data:
        total_union = overlap_data.get("total_union")
        if total_union:
            out["union_of_all_variants.csv"] = pd.Series(sorted(total_union)).to_csv(index=False, header=["variant_id"])
        common = overlap_data.get("common_in_all")
        if common is not None:
            out["variants_common_all_files.csv"] = pd.Series(sorted(common)).to_csv(index=False, header=["variant_id"])
        for name, ids in overlap_data.get("unique_per_file", {}).items():
            safe = name.replace("/", "_").replace(" ", "_")
            out[f"variants_unique_to_{safe}.csv"] = pd.Series(sorted(ids)).to_csv(index=False, header=["variant_id"])
    if mismatch_df is not None and not mismatch_df.empty:
        out["column_mismatches.csv"] = mismatch_df.to_csv(index=False)
    if header_table is not None and not header_table.empty:
        out["header_comparison.csv"] = header_table.to_csv(index=False)
    if missing_pairs:
        for (a, b), ids in missing_pairs.items():
            if ids:
                safe_a, safe_b = a.replace("/", "_").replace(" ", "_"), b.replace("/", "_").replace(" ", "_")
                out[f"missing_in_{safe_b}_but_in_{safe_a}.csv"] = pd.Series(sorted(ids)).to_csv(index=False, header=["variant_id"])
    if clinical_mismatch_df is not None and not clinical_mismatch_df.empty:
        out["clinical_concordance_mismatches.csv"] = clinical_mismatch_df.to_csv(index=False)
    return out


def _escape_xml(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _append_clinical_validation_pdf(story: list, clinical: dict, styles: dict, h1, h2, body) -> None:
    """Separate page: Cloud vs On-Prem executive validation."""
    from clinical_concordance import validation_summary_line

    story.append(PageBreak())
    story.append(Paragraph("Executive validation — Cloud vs On-Prem clinical concordance", h1))
    story.append(Spacer(1, 0.15 * inch))

    if clinical.get("error"):
        story.append(Paragraph(_escape_xml(f"Validation could not be completed: {clinical['error']}"), body))
        return

    story.append(Paragraph("<b>Purpose (plain language)</b>", h2))
    snap0 = clinical.get("config_snapshot") or {}
    scope_gene = "all genes in scope (after filters)" if snap0.get("compare_all_genes") else "one selected gene"
    story.append(Paragraph(
        "This section compares matching variant rows between the Cloud pipeline output and On-Prem for "
        f"<b>{_escape_xml(scope_gene)}</b>. Aligned fields are compared in bulk (vectorized) so leadership can "
        "see whether clinical-grade annotations agree between environments.",
        body,
    ))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("<b>What this means</b>", h2))
    conc = clinical.get("concordance_pct")
    story.append(Paragraph(
        _escape_xml(
            f"Overall concordance is {conc:.2f}% across all cell-level tests in scope. "
            "Strict columns (e.g. cDNA / protein / HGVS genomic) must match exactly; other numeric fields may differ within the configured tolerance."
        ),
        body,
    ))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Scope and methodology</b>", h2))
    snap = clinical.get("config_snapshot") or {}
    gene_line = (
        "All genes (gene filter off)"
        if snap.get("compare_all_genes")
        else f"Gene: <b>{_escape_xml(str(snap.get('gene', '—')))}</b> ({_escape_xml(str(snap.get('gene_col', '') or '—'))})"
    )
    col_line = (
        "Columns compared: user-selected subset only"
        if snap.get("compare_columns_filter")
        else "Columns compared: all shared annotation columns (except keys / gene / VARIANT_LOCATION)"
    )
    bullets = [
        gene_line,
        f"Files: {_escape_xml(str(snap.get('label_cloud', 'Cloud')))} vs {_escape_xml(str(snap.get('label_linc', 'On-Prem')))}",
        "Row match: CHROM, START, REF, ALT, and ENS_TRANS_ID (inner join; one row per key after de-duplication).",
        f"On-target filter: VARIANT_LOCATION = ONTARGET — <b>{'yes' if snap.get('restrict_ontarget') else 'no'}</b>",
        f"Numeric tolerance: ±{_escape_xml(str(snap.get('tolerance_pct', 10)))}% with Cloud as reference.",
        f"Strict columns (zero tolerance): {_escape_xml(', '.join(clinical.get('strict_columns_used') or []) or '—')}",
        col_line,
        f"Mismatch export cap: first {_escape_xml(str(snap.get('max_mismatch_rows_stored', 250000)))} detail rows (totals still exact).",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("<b>Limitations</b>", h2))
    lim = [
        "Rows must appear in both files after filters (inner join on variant + transcript).",
        "If a single gene was chosen, only that gene is in scope; “all genes” uses every row after on-target and other filters.",
        "Variants present in only one file are excluded from paired cell tests.",
        "Duplicate keys in a source file were collapsed (first row kept); see KPI duplicate counts.",
        "Very large runs should use column subsets or one gene so end-to-end time stays predictable (e.g. under a few minutes excluding upload).",
    ]
    for L in lim:
        story.append(Paragraph(f"• {_escape_xml(L)}", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Key results</b>", h2))
    kpi = [
        ["Metric", "Value"],
        ["Concordance (%)", f"{clinical.get('concordance_pct', 0):.4f}"],
        ["Total cell-level tests", str(clinical.get("total_tests", 0))],
        ["Matched variant-transcript rows", str(clinical.get("n_rows_matched", 0))],
        ["Rows after filters (Cloud)", str(clinical.get("n_rows_cloud_after_filters", 0))],
        ["Rows after filters (On-Prem)", str(clinical.get("n_rows_linc_after_filters", 0))],
        ["Duplicate rows dropped (Cloud)", str(clinical.get("n_duplicate_rows_dropped_cloud", 0))],
        ["Duplicate rows dropped (On-Prem)", str(clinical.get("n_duplicate_rows_dropped_linc", 0))],
        ["All mismatches", str(clinical.get("n_failed", 0))],
        ["Strict column mismatches", str(clinical.get("n_failed_strict", 0))],
        ["Non-strict mismatches", str(clinical.get("n_failed_non_strict", 0))],
    ]
    if clinical.get("n_failed_strict", 0) > 0:
        kpi.append(["Strict-field status", "Requires review"])
    else:
        kpi.append(["Strict-field status", "No strict mismatches"])

    kt = Table(kpi, colWidths=[3.2 * inch, 2.8 * inch], repeatRows=1)
    kt.setStyle(TableStyle(_table_style_modern()))
    story.append(kt)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Validation summary</b>", h2))
    story.append(Paragraph(_escape_xml(validation_summary_line(clinical)), body))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        f"<i>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — for management and client review; not a regulatory sign-off.</i>",
        body,
    ))

    mdf = clinical.get("mismatch_df")
    if mdf is not None and not mdf.empty and hasattr(mdf, "head"):
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("<b>Detailed findings (first 40 rows)</b>", h2))
        story.append(Paragraph(
            "Full mismatch list is available in <b>clinical_concordance_mismatches.csv</b> from the dashboard.",
            body,
        ))
        disp_cols = [c for c in ("label", "column", "strict_column", "Cloud_value", "On-Prem_value") if c in mdf.columns]
        if not disp_cols:
            disp_cols = list(mdf.columns[:8])
        sub = mdf[disp_cols].head(40)
        data = _df_to_table_data(sub, max_rows=40)
        nc = max(len(data[0]), 1)
        mw = min(6.5 / nc, 1.6)
        for fl in _safe_table(data, [mw * inch] * nc, font_size=6, rows_per_chunk=40):
            story.append(fl)


def generate_pdf_report(
    qc_score: int,
    qc_status: str,
    n_files: int,
    total_records: int,
    total_unique: int,
    common_count: int,
    record_metrics: pd.DataFrame,
    header_similarity: float,
    header_table: pd.DataFrame,
    overlap_data: dict,
    mismatch_count: int,
    dup_analysis: dict,
    management_summary: dict,
    filename_prefix: str = "QC_Report",
    clinical_concordance: Optional[dict[str, Any]] = None,
) -> bytes:
    """
    Generate a professional PDF QC report with executive summary and optional Cloud vs On-Prem section.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="ExecTitle",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=6,
        textColor=_CLR_HEADER,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        name="ExecSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=14,
        textColor=colors.HexColor("#64748b"),
    )
    h2_style = ParagraphStyle(
        name="ExecH2",
        parent=styles["Heading2"],
        fontSize=13,
        spaceAfter=8,
        spaceBefore=6,
        textColor=_CLR_HEADER,
        fontName="Helvetica-Bold",
    )
    body = ParagraphStyle(
        name="ExecBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # Cover-style title block
    story.append(Paragraph("VariMAT Quality &amp; Validation Report", title_style))
    story.append(Paragraph("Suitable for executive, product, and client review", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    story.append(Spacer(1, 0.2 * inch))

    # Executive summary bullets
    story.append(Paragraph("Executive summary", h2_style))
    es_lines = [
        f"{n_files} VariMAT file(s) compared; {total_records:,} total records; {total_unique:,} unique variants (union).",
        f"{common_count:,} variants are common to all files.",
        f"Header similarity across files: {header_similarity}%.",
        f"Overall QC score: {qc_score}/100 — {qc_status}.",
    ]
    clin = clinical_concordance
    if clin and not clin.get("error") and clin.get("total_tests"):
        es_lines.append(
            f"Cloud vs On-Prem concordance (selected scope): {clin.get('concordance_pct', 0):.2f}% "
            f"({clin.get('n_failed_strict', 0)} strict-field mismatch(es))."
        )
    elif clin and clin.get("error"):
        es_lines.append(f"Cloud vs On-Prem validation: not included — {clin.get('error')}.")
    for line in es_lines:
        story.append(Paragraph(f"• {_escape_xml(line)}", body))
    story.append(Spacer(1, 0.25 * inch))

    # Management Summary table
    story.append(Paragraph("Management summary", h2_style))
    mgmt = [
        ["Metric", "Value"],
        ["Files compared", str(n_files)],
        ["Total records (all files)", str(total_records)],
        ["Unique variants (union)", str(total_unique)],
        ["Variants common in all files", str(common_count)],
        ["Variant consistency", management_summary.get("variant_consistency", "N/A")],
        ["QC Score", f"{qc_score} — {qc_status}"],
    ]
    t = Table(mgmt, colWidths=[3 * inch, 3 * inch], repeatRows=1)
    t.setStyle(TableStyle(_table_style_modern()))
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("QC score detail", h2_style))
    story.append(Paragraph(f"<b>Score: {qc_score}/100</b> — {_escape_xml(qc_status)}", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Record-level metrics", h2_style))
    if record_metrics is not None and not record_metrics.empty:
        data = _df_to_table_data(record_metrics)
        tw = Table(data, colWidths=[1.2 * inch] * len(data[0]), repeatRows=1)
        tw.setStyle(TableStyle(_table_style_modern()))
        story.append(tw)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Header consistency", h2_style))
    story.append(Paragraph(f"Header similarity across files: {header_similarity}%", body))
    if header_table is not None and not header_table.empty:
        data = _df_to_table_data(header_table, max_rows=30)
        col_w = 1.5 * inch if len(data[0]) <= 4 else 1.0 * inch
        th = Table(data, colWidths=[col_w] * len(data[0]), repeatRows=1)
        th.setStyle(TableStyle(_table_style_modern() + [("FONTSIZE", (0, 0), (-1, -1), 6)]))
        story.append(th)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Variant overlap", h2_style))
    if overlap_data:
        story.append(Paragraph(f"Variants common in all files: {overlap_data.get('common_count', 0)}", body))
        for name, ids in overlap_data.get("unique_per_file", {}).items():
            story.append(Paragraph(f"Unique to {_escape_xml(str(name))}: {len(ids)}", body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Column value mismatches (multi-file QC)", h2_style))
    story.append(Paragraph(f"Total mismatch records detected: {mismatch_count}", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Duplicate variant summary", h2_style))
    if dup_analysis:
        for fname, d in dup_analysis.items():
            story.append(Paragraph(
                f"{_escape_xml(str(fname))}: {d.get('n_duplicated_variants', 0)} variants with duplicate rows",
                body,
            ))
    story.append(Spacer(1, 0.15 * inch))

    if clinical_concordance:
        _append_clinical_validation_pdf(story, clinical_concordance, styles, title_style, h2_style, body)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def get_pdf_filename() -> str:
    """Return default PDF filename with date."""
    return f"QC_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"


# CONCORDANCE — standalone concordance PDF filename
def get_concordance_pdf_filename() -> str:
    return f"cloud_onprem_concordance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"


# CONCORDANCE — standalone concordance CSV exports
def generate_concordance_csv_exports(concordance_result: dict[str, Any]) -> dict[str, bytes]:
    """Return dict of {filename: utf-8 CSV bytes} for concordance-specific downloads."""
    out: dict[str, bytes] = {}

    mdf = concordance_result.get("mismatch_df")
    if mdf is not None and not mdf.empty:
        out["concordance_mismatches.csv"] = mdf.head(250_000).to_csv(index=False).encode("utf-8")
    else:
        out["concordance_mismatches.csv"] = b"No mismatches detected.\n"

    match_df = concordance_result.get("match_df")
    if match_df is not None and not match_df.empty:
        out["concordance_matches.csv"] = match_df.head(250_000).to_csv(index=False).encode("utf-8")
    else:
        out["concordance_matches.csv"] = b"No match data available.\n"

    col_pcts = concordance_result.get("col_concordance_pct", {})
    col_counts = concordance_result.get("col_mismatch_counts", {})
    col_totals = concordance_result.get("col_total_tests", {})
    if col_pcts:
        rows = []
        for col, pct in sorted(col_pcts.items(), key=lambda x: x[1]):
            rows.append({
                "column": col,
                "concordance_pct": round(pct, 4),
                "n_mismatches": col_counts.get(col, 0),
                "n_total_tests": col_totals.get(col, 0),
                "status": "PASS" if pct == 100.0 else ("REVIEW" if pct >= 95.0 else "FAIL"),
            })
        out["concordance_per_column_summary.csv"] = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    else:
        out["concordance_per_column_summary.csv"] = b"No per-column data available.\n"

    excl = concordance_result.get("excluded_cols", [])
    summary_row = {
        "overall_concordance_pct": concordance_result.get("concordance_pct", 0),
        "n_matched_rows": concordance_result.get("n_rows_matched", 0),
        "n_perfect_match_rows": concordance_result.get("n_perfect_match_rows", 0),
        "n_mismatch_rows": concordance_result.get("n_mismatch_rows", 0),
        "n_failed_strict": concordance_result.get("n_failed_strict", 0),
        "n_failed_non_strict": concordance_result.get("n_failed_non_strict", 0),
        "total_tests": concordance_result.get("total_tests", 0),
        "columns_compared": len(concordance_result.get("columns_compared", [])),
        "excluded_cols": ", ".join(excl) if excl else "",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out["concordance_summary.csv"] = pd.DataFrame([summary_row]).to_csv(index=False).encode("utf-8")

    return out


# CONCORDANCE — standalone concordance PDF report
def generate_concordance_pdf_report(
    concordance_result: dict[str, Any],
    label_cloud: str = "Cloud",
    label_onprem: str = "On-Prem",
) -> bytes:
    """Produce a standalone multi-page concordance validation PDF. Returns bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        name="ConcH1", parent=styles["Heading1"],
        fontSize=18, spaceAfter=8, spaceBefore=4, textColor=_CLR_HEADER, fontName="Helvetica-Bold",
    )
    h2 = ParagraphStyle(
        name="ConcH2", parent=styles["Heading2"],
        fontSize=12, spaceAfter=6, spaceBefore=10, textColor=_CLR_HEADER, fontName="Helvetica-Bold",
    )
    h3 = ParagraphStyle(
        name="ConcH3", parent=styles["Heading3"],
        fontSize=10, spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#334155"), fontName="Helvetica-Bold",
    )
    body = ParagraphStyle(
        name="ConcBody", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=colors.HexColor("#334155"),
    )
    body_small = ParagraphStyle(
        name="ConcBodySm", parent=body, fontSize=8, leading=11,
    )
    bullet = ParagraphStyle(
        name="ConcBullet", parent=body, leftIndent=14, bulletIndent=4,
    )
    note_style = ParagraphStyle(
        name="ConcNote", parent=body, fontSize=8, leading=10,
        textColor=colors.HexColor("#64748b"), fontName="Helvetica-Oblique",
    )

    story: list = []
    r = concordance_result
    n_strict = r.get("n_failed_strict", 0)
    concordance = r.get("concordance_pct", 0)
    n_matched = r.get("n_rows_matched", 0)
    n_perfect = r.get("n_perfect_match_rows", 0)
    n_mismatch_rows = r.get("n_mismatch_rows", 0)
    total_tests = r.get("total_tests", 0)
    n_cols = len(r.get("columns_compared", []))
    excl = r.get("excluded_cols", [])
    strict_used = r.get("strict_columns_used", [])
    snap = r.get("config_snapshot") or {}

    # Determine overall verdict once
    if n_strict == 0 and concordance >= 99.0:
        verdict = "PASS"
        verdict_color = colors.HexColor("#059669")
        verdict_long = (
            "The files are concordant. No strict clinical field mismatches were detected "
            f"and overall concordance is {concordance:.4f}%, which meets the ≥ 99% threshold. "
            "These files are validated for upload."
        )
    elif n_strict == 0 and concordance >= 95.0:
        verdict = "REVIEW"
        verdict_color = colors.HexColor("#d97706")
        verdict_long = (
            f"No strict clinical field mismatches were found, but overall concordance "
            f"is {concordance:.2f}%, which is below the 99% PASS threshold. "
            "Non-strict annotation differences exist and should be investigated before upload. "
            "See the per-column breakdown and mismatch detail for specifics."
        )
    else:
        verdict = "FAIL"
        verdict_color = colors.HexColor("#dc2626")
        if n_strict > 0:
            verdict_long = (
                f"{n_strict:,} strict clinical field mismatch(es) were detected in fields that require "
                "exact agreement (CDNA_CHG, AA_CHG, HGVS). These are critical for clinical reporting "
                "and must be resolved before the files can be considered concordant. "
                "Do NOT upload until all strict mismatches are resolved."
            )
        else:
            verdict_long = (
                f"Overall concordance is {concordance:.2f}%, which is below the 95% threshold. "
                "A significant number of annotation fields differ between environments. "
                "Review the per-column breakdown to identify which columns are driving the gap."
            )

    verdict_style = ParagraphStyle(
        name="ConcVerdict", parent=body, fontSize=13, leading=16,
        textColor=verdict_color, fontName="Helvetica-Bold",
    )

    # ────────────────────────────────────────────────────────────────
    # PAGE 1 — Executive Summary
    # ────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        f"{_escape_xml(label_cloud)} vs {_escape_xml(label_onprem)} "
        "Concordance Validation Report", h1))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        note_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(f"Validation Result:  {verdict}", verdict_style))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(verdict_long, body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>What is this report?</b>", h2))
    story.append(Paragraph(
        "This report compares the output of two variant annotation pipelines — "
        f"<b>{_escape_xml(label_cloud)}</b> and <b>{_escape_xml(label_onprem)}</b> — "
        "to verify that they produce identical or near-identical clinical annotations "
        "for the same set of variants. This is a product validation concordance check. "
        "Every variant present in both files is matched by its genomic coordinates and "
        "transcript, then every shared annotation column is compared cell-by-cell.",
        body))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("<b>Key Performance Indicators</b>", h2))
    kpi = [
        ["Metric", "Value", "Interpretation"],
        ["Overall Concordance",
         f"{concordance:.4f}%",
         "≥ 99% = PASS, 95–99% = REVIEW, < 95% = FAIL"],
        ["Perfectly Matched Rows",
         f"{n_perfect:,} of {n_matched:,}",
         "Rows where every compared column matched exactly"],
        ["Rows with Any Mismatch",
         f"{n_mismatch_rows:,}",
         "Rows where at least one column differed"],
        ["Strict Field Mismatches",
         f"{n_strict:,}",
         "CDNA_CHG / AA_CHG / HGVS — must be zero for PASS"],
        ["Non-strict Mismatches",
         f"{r.get('n_failed_non_strict', 0):,}",
         "Numeric fields within tolerance are acceptable"],
        ["Total Cell-level Tests",
         f"{total_tests:,}",
         f"{n_matched:,} rows × {n_cols} columns"],
        ["Columns Compared",
         str(n_cols),
         "Shared annotation columns (excluding keys and EI_TOTAL)"],
        ["Columns Excluded",
         ", ".join(excl) if excl else "—",
         "EI_TOTAL: known to vary between environments"],
    ]
    kt = Table(kpi, colWidths=[1.7 * inch, 1.5 * inch, 3.5 * inch], repeatRows=1)
    kt.setStyle(TableStyle(_table_style_modern() + [("FONTSIZE", (0, 0), (-1, -1), 7)]))
    story.append(kt)

    # ────────────────────────────────────────────────────────────────
    # PAGE 2 — How to Read This Report / Methodology
    # ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("How to Read This Report", h1))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph('<b>What does "concordance" mean?</b>', h3))
    story.append(Paragraph(
        "Concordance is the percentage of individual cell-level comparisons that agree "
        "between the two files. For example, if 100 variant rows are matched and 50 columns "
        "are compared per row, there are 5,000 cell-level tests. If 4,990 agree, "
        "concordance is 99.80%. A higher concordance means the two pipelines produce "
        "more similar results.",
        body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("<b>Validation thresholds</b>", h3))
    thresh = [
        ["Verdict", "Condition", "Action Required"],
        ["PASS",
         "Concordance ≥ 99% AND zero strict field mismatches",
         "Files are validated. Safe to upload."],
        ["REVIEW",
         "Concordance ≥ 95% AND zero strict field mismatches",
         "Investigate non-strict differences before upload."],
        ["FAIL",
         "Concordance < 95% OR any strict field mismatch",
         "Do NOT upload. Resolve mismatches first."],
    ]
    tt = Table(thresh, colWidths=[0.9 * inch, 3.0 * inch, 2.8 * inch], repeatRows=1)
    tt.setStyle(TableStyle(_table_style_modern() + [("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(tt)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Tolerance rules by column type</b>", h3))
    story.append(Paragraph(
        "Not all columns are compared the same way. Clinical-critical fields must "
        "match exactly, while numeric annotation scores allow small rounding differences.",
        body))
    story.append(Spacer(1, 0.06 * inch))
    tol_rules = [
        ["Column Type", "Examples", "Tolerance", "Rationale"],
        ["Strict (clinical)",
         ", ".join(strict_used[:4]) if strict_used else "CDNA_CHG, AA_CHG, HGVS*",
         "Zero — exact string match",
         "These appear on clinical reports and must agree exactly."],
        ["VarTk score",
         "VARTK_SCORE, VarTk_score",
         "±5% of Cloud value",
         "Tighter than default; small float rounding is acceptable."],
        ["Other numeric",
         "All remaining annotation fields",
         f"±{_escape_xml(str(snap.get('tolerance_pct', 10)))}% of Cloud value",
         "Allows minor numeric differences from rounding/precision."],
        ["Excluded",
         ", ".join(excl) if excl else "EI_TOTAL",
         "Not compared",
         "Known to vary between environments; excluded by design."],
    ]
    trt = Table(tol_rules, colWidths=[1.2 * inch, 1.8 * inch, 1.6 * inch, 2.1 * inch], repeatRows=1)
    trt.setStyle(TableStyle(_table_style_modern() + [("FONTSIZE", (0, 0), (-1, -1), 7)]))
    story.append(trt)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("<b>Row matching methodology</b>", h3))
    meth_bullets = [
        "Each variant row is identified by five key columns: <b>CHROM, START, REF, ALT, ENS_TRANS_ID</b>.",
        "An inner join matches rows present in both files. Variants only in one file are excluded.",
        "If a file has duplicate keys (same CHROM+START+REF+ALT+ENS_TRANS_ID), only the first row is kept.",
        f"On-target filter: only rows where VARIANT_LOCATION = ONTARGET are included — "
        f"<b>{'enabled' if snap.get('restrict_ontarget') else 'disabled'}</b>.",
        "After matching, every shared annotation column (except keys, gene, VARIANT_LOCATION, and excluded columns) "
        "is compared cell-by-cell using the tolerance rules above.",
        "Empty/NA values in both files are treated as matching.",
    ]
    for b in meth_bullets:
        story.append(Paragraph(f"• {b}", bullet))

    # ────────────────────────────────────────────────────────────────
    # PAGE 3 — File & Row Statistics
    # ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("File &amp; Row Statistics", h1))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "This section shows how many rows from each file survived the filtering and "
        "de-duplication steps before being compared.",
        body))
    story.append(Spacer(1, 0.1 * inch))

    n_cloud_after = r.get("n_rows_cloud_after_filters", 0)
    n_onprem_after = r.get("n_rows_linc_after_filters", 0)
    n_dup_cloud = r.get("n_duplicate_rows_dropped_cloud", 0)
    n_dup_onprem = r.get("n_duplicate_rows_dropped_linc", 0)

    file_stats = [
        ["Metric", _escape_xml(label_cloud), _escape_xml(label_onprem)],
        ["Rows after on-target filter", f"{n_cloud_after:,}", f"{n_onprem_after:,}"],
        ["Duplicate rows dropped", f"{n_dup_cloud:,}", f"{n_dup_onprem:,}"],
        ["Unique rows available for matching",
         f"{n_cloud_after - n_dup_cloud:,}",
         f"{n_onprem_after - n_dup_onprem:,}"],
    ]
    fst = Table(file_stats, colWidths=[2.8 * inch, 1.8 * inch, 1.8 * inch], repeatRows=1)
    fst.setStyle(TableStyle(_table_style_modern()))
    story.append(fst)
    story.append(Spacer(1, 0.15 * inch))

    match_rate = (n_matched / max(min(n_cloud_after - n_dup_cloud,
                                       n_onprem_after - n_dup_onprem), 1)) * 100
    join_stats = [
        ["Metric", "Value", "Interpretation"],
        ["Rows matched (inner join)", f"{n_matched:,}",
         "Variant+transcript keys found in both files"],
        ["Match rate",
         f"{match_rate:.1f}%",
         "Matched rows / smaller file's unique rows"],
        ["Perfectly concordant rows", f"{n_perfect:,}",
         "Every column matched across all compared fields"],
        ["Rows with at least one mismatch", f"{n_mismatch_rows:,}",
         "At least one annotation column differs"],
    ]
    jst = Table(join_stats, colWidths=[2.2 * inch, 1.3 * inch, 3.2 * inch], repeatRows=1)
    jst.setStyle(TableStyle(_table_style_modern() + [("FONTSIZE", (0, 0), (-1, -1), 7)]))
    story.append(jst)
    story.append(Spacer(1, 0.15 * inch))

    if n_matched > 0 and n_mismatch_rows == 0:
        story.append(Paragraph(
            f"<b>All {n_matched:,} matched rows are perfectly concordant across "
            f"all {n_cols} compared columns.</b> This means the two pipelines produced "
            "identical annotations for every variant in scope.",
            body))
    elif n_matched > 0:
        pct_perfect = 100.0 * n_perfect / n_matched
        story.append(Paragraph(
            f"<b>{pct_perfect:.1f}%</b> of matched rows ({n_perfect:,} of {n_matched:,}) "
            f"are perfectly concordant across all {n_cols} columns. "
            f"The remaining {n_mismatch_rows:,} rows have at least one column that differs. "
            "See the per-column breakdown to understand which fields drive these differences.",
            body))

    # ────────────────────────────────────────────────────────────────
    # PAGE 4+ — Per-Column Concordance (chunked tables)
    # ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Per-Column Concordance Breakdown", h1))
    story.append(Spacer(1, 0.08 * inch))

    col_pcts = r.get("col_concordance_pct", {})
    col_counts = r.get("col_mismatch_counts", {})
    col_totals = r.get("col_total_tests", {})

    if col_pcts:
        n_perfect_cols = sum(1 for p in col_pcts.values() if p == 100.0)
        n_review_cols = sum(1 for p in col_pcts.values() if 95.0 <= p < 100.0)
        n_fail_cols = sum(1 for p in col_pcts.values() if p < 95.0)

        story.append(Paragraph(
            f"Of <b>{len(col_pcts)}</b> columns compared: "
            f"<b>{n_perfect_cols}</b> are 100% concordant (PASS), "
            f"<b>{n_review_cols}</b> are 95–99.99% (REVIEW), "
            f"and <b>{n_fail_cols}</b> are below 95% (FAIL).",
            body))
        if n_perfect_cols == len(col_pcts):
            story.append(Paragraph(
                "<b>Every single column matched perfectly across all rows.</b> "
                "The two pipelines produce identical annotations.",
                body))
        story.append(Spacer(1, 0.1 * inch))

        col_data = [["Column Name", "Concordance %", "Mismatches", "Rows Tested", "Status"]]
        for col, pct in sorted(col_pcts.items(), key=lambda x: x[1]):
            status = "PASS" if pct == 100.0 else ("REVIEW" if pct >= 95.0 else "FAIL")
            col_data.append([
                col, f"{pct:.4f}", str(col_counts.get(col, 0)),
                str(col_totals.get(col, 0)), status,
            ])

        col_widths = [2.2 * inch, 1.2 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch]
        for fl in _safe_table(col_data, col_widths, font_size=7, rows_per_chunk=42):
            story.append(fl)

        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph(
            "<b>How to interpret:</b> A column with 100% concordance means every cell in that column "
            "matched between Cloud and On-Prem for all matched rows. A column below 100% has differences — "
            "the mismatch count tells you how many cells differed. Strict clinical columns must be 100%; "
            "non-strict columns allow numeric tolerance.",
            note_style))
    else:
        story.append(Paragraph("No per-column data available.", body))

    # ────────────────────────────────────────────────────────────────
    # Strict Field Analysis
    # ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Strict Clinical Field Analysis", h1))
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph(
        "Strict fields are clinical-critical annotations that appear on patient reports: "
        "<b>CDNA_CHG</b> (cDNA change), <b>AA_CHG</b> (amino acid change), and "
        "<b>HGVS genomic notation</b>. These must match exactly between environments — "
        "no numeric tolerance is applied.",
        body))
    story.append(Spacer(1, 0.08 * inch))

    if strict_used:
        story.append(Paragraph(
            f"Strict columns detected in data: <b>{_escape_xml(', '.join(strict_used))}</b>",
            body))
    else:
        story.append(Paragraph(
            "No strict columns were detected in the shared columns of these files. "
            "This may indicate the files use different column naming.",
            body))
    story.append(Spacer(1, 0.1 * inch))

    strict_vars = r.get("strict_fail_variants", [])
    if n_strict == 0:
        story.append(Paragraph(
            "<b>Result: No strict clinical field failures.</b> "
            "All clinical report fields (cDNA change, amino acid change, HGVS notation) "
            "agree exactly between Cloud and On-Prem for every matched variant. "
            "This is the required condition for clinical concordance.",
            body))
    else:
        story.append(Paragraph(
            f"<b>Result: {n_strict:,} strict field mismatch(es) detected.</b> "
            f"These affect {len(strict_vars)} unique variant(s). Each mismatch means a clinical "
            "report field differs between environments — this must be investigated and resolved.",
            body))
        story.append(Spacer(1, 0.1 * inch))

        if strict_vars:
            story.append(Paragraph(
                f"Variants with strict failures (first {min(len(strict_vars), 300)}):", h3))
            sv_data = [["#", "Variant Key (CHROM:START:REF:ALT:ENS_TRANS_ID)"]]
            for i, vk in enumerate(strict_vars[:300], 1):
                sv_data.append([str(i), _escape_xml(str(vk))])
            for fl in _safe_table(sv_data,
                                  [0.5 * inch, 6.2 * inch],
                                  font_size=6, rows_per_chunk=50):
                story.append(fl)

    # ────────────────────────────────────────────────────────────────
    # Mismatch Detail (chunked)
    # ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Mismatch Detail", h1))
    story.append(Spacer(1, 0.08 * inch))

    mdf = r.get("mismatch_df")
    if mdf is not None and not mdf.empty:
        total_mm = r.get("n_mismatch_rows", len(mdf))
        show_n = min(len(mdf), 500)
        story.append(Paragraph(
            f"This table shows the first <b>{show_n}</b> individual cell-level mismatches "
            f"out of <b>{r.get('n_failed', total_mm):,}</b> total. Each row represents one "
            "cell comparison that did not agree between Cloud and On-Prem. "
            "The full list is available in the <b>concordance_mismatches.csv</b> download.",
            body))
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph(
            "<b>Reading the table:</b> 'column' is the annotation field that differed. "
            f"'{_escape_xml(label_cloud)}_value' and '{_escape_xml(label_onprem)}_value' show "
            "the actual values. 'strict_column' indicates whether this is a zero-tolerance clinical field.",
            note_style))
        story.append(Spacer(1, 0.08 * inch))

        disp_cols = [c for c in ("variant_key", "column", "strict_column",
                                 f"{label_cloud}_value", f"{label_onprem}_value")
                     if c in mdf.columns]
        if not disp_cols:
            disp_cols = list(mdf.columns[:6])
        sub = mdf[disp_cols].head(show_n)
        data = [sub.columns.tolist()] + sub.fillna("").astype(str).values.tolist()
        nc = len(data[0])
        # Proportional widths for the 5 standard columns
        if nc == 5:
            cw = [1.6 * inch, 1.1 * inch, 0.7 * inch, 1.7 * inch, 1.7 * inch]
        else:
            mw = min(6.8 / max(nc, 1), 1.6)
            cw = [mw * inch] * nc
        for fl in _safe_table(data, cw, font_size=6, rows_per_chunk=45):
            story.append(fl)
    else:
        story.append(Paragraph(
            "<b>No mismatches were detected.</b> Every cell-level comparison between "
            f"{_escape_xml(label_cloud)} and {_escape_xml(label_onprem)} agreed across "
            f"all {n_cols} compared columns and all {n_matched:,} matched rows.",
            body))

    # ────────────────────────────────────────────────────────────────
    # Match Summary
    # ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Match Summary — Perfectly Concordant Rows", h1))
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph(
        f"<b>{n_perfect:,}</b> rows matched perfectly across all <b>{n_cols}</b> "
        f"compared columns. This means every single annotation value agreed between "
        f"{_escape_xml(label_cloud)} and {_escape_xml(label_onprem)} for these variants.",
        body))
    story.append(Spacer(1, 0.06 * inch))

    if n_matched > 0:
        pct_perfect = 100.0 * n_perfect / n_matched
        story.append(Paragraph(
            f"Perfect match rate: <b>{pct_perfect:.2f}%</b> of all matched rows.",
            body))
    story.append(Spacer(1, 0.1 * inch))

    match_df = r.get("match_df")
    if match_df is not None and not match_df.empty:
        show_match = min(len(match_df), 100)
        story.append(Paragraph(f"Sample of first {show_match} matched rows:", h3))
        mdata = [match_df.columns.tolist()] + match_df.head(show_match).fillna("").astype(str).values.tolist()
        mnc = len(mdata[0])
        if mnc == 6:
            mcw = [1.0 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch, 1.8 * inch, 1.0 * inch]
        else:
            mmw = min(6.8 / max(mnc, 1), 1.4)
            mcw = [mmw * inch] * mnc
        for fl in _safe_table(mdata, mcw, font_size=6, rows_per_chunk=45):
            story.append(fl)

    # ────────────────────────────────────────────────────────────────
    # FINAL PAGE — Conclusion & Recommendation
    # ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Conclusion &amp; Recommendation", h1))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(f"Overall Verdict:  {verdict}", verdict_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(verdict_long, body))
    story.append(Spacer(1, 0.15 * inch))

    # Summary stats recap as a compact table
    summary_recap = [
        ["Metric", "Value"],
        ["Overall Concordance", f"{concordance:.4f}%"],
        ["Matched Rows", f"{n_matched:,}"],
        ["Perfectly Concordant Rows", f"{n_perfect:,}"],
        ["Rows with Mismatches", f"{n_mismatch_rows:,}"],
        ["Strict Field Mismatches", f"{n_strict:,}"],
        ["Non-strict Mismatches", f"{r.get('n_failed_non_strict', 0):,}"],
        ["Columns Compared", str(n_cols)],
        ["Verdict", verdict],
    ]
    srt = Table(summary_recap, colWidths=[3.0 * inch, 3.0 * inch], repeatRows=1)
    srt.setStyle(TableStyle(_table_style_modern()))
    story.append(srt)
    story.append(Spacer(1, 0.2 * inch))

    # Actionable next steps based on verdict
    story.append(Paragraph("<b>Recommended Next Steps</b>", h2))
    if verdict == "PASS":
        steps = [
            "Files are validated. Proceed with upload to the production system.",
            "Archive this report alongside the uploaded files for audit trail.",
            "No further investigation of annotation differences is required.",
        ]
    elif verdict == "REVIEW":
        steps = [
            "Download concordance_per_column_summary.csv to identify which columns have < 100% concordance.",
            "Download concordance_mismatches.csv and review the specific values that differ.",
            "Determine whether the differences are acceptable rounding/precision changes "
            "or indicate a genuine pipeline discrepancy.",
            "If differences are acceptable, document the rationale and proceed with upload.",
            "If differences are unexpected, escalate to the pipeline engineering team.",
        ]
    else:
        steps = [
            "Do NOT upload these files until all strict field mismatches are resolved.",
            "Download concordance_mismatches.csv and filter to strict_column = True.",
            "For each strict mismatch, identify the root cause in the pipeline.",
            "After fixes, re-run both pipelines and generate a new concordance report.",
            "Escalate to the clinical team if HGVS or cDNA changes are affected.",
        ]
    for i, s in enumerate(steps, 1):
        story.append(Paragraph(f"{i}. {_escape_xml(s)}", bullet))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        f"<i>Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
        "for management, product, and clinical review. Not a regulatory sign-off.</i>",
        note_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
