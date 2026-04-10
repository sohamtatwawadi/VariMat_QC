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
    """Separate page: Cloud vs LinC executive validation."""
    from clinical_concordance import validation_summary_line

    story.append(PageBreak())
    story.append(Paragraph("Executive validation — Cloud vs LinC clinical concordance", h1))
    story.append(Spacer(1, 0.15 * inch))

    if clinical.get("error"):
        story.append(Paragraph(_escape_xml(f"Validation could not be completed: {clinical['error']}"), body))
        return

    story.append(Paragraph("<b>Purpose (plain language)</b>", h2))
    snap0 = clinical.get("config_snapshot") or {}
    scope_gene = "all genes in scope (after filters)" if snap0.get("compare_all_genes") else "one selected gene"
    story.append(Paragraph(
        "This section compares matching variant rows between the Cloud pipeline output and LinC for "
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
        f"Files: {_escape_xml(str(snap.get('label_cloud', 'Cloud')))} vs {_escape_xml(str(snap.get('label_linc', 'LinC')))}",
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
        ["Rows after filters (LinC)", str(clinical.get("n_rows_linc_after_filters", 0))],
        ["Duplicate rows dropped (Cloud)", str(clinical.get("n_duplicate_rows_dropped_cloud", 0))],
        ["Duplicate rows dropped (LinC)", str(clinical.get("n_duplicate_rows_dropped_linc", 0))],
        ["All mismatches", str(clinical.get("n_failed", 0))],
        ["Strict column mismatches", str(clinical.get("n_failed_strict", 0))],
        ["Non-strict mismatches", str(clinical.get("n_failed_non_strict", 0))],
    ]
    if clinical.get("n_failed_strict", 0) > 0:
        kpi.append(["Strict-field status", "Requires review"])
    else:
        kpi.append(["Strict-field status", "No strict mismatches"])

    kt = Table(kpi, colWidths=[3.2 * inch, 2.8 * inch])
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
        disp_cols = [c for c in ("label", "column", "strict_column", "Cloud_value", "LinC_value") if c in mdf.columns]
        if not disp_cols:
            disp_cols = list(mdf.columns[:8])
        sub = mdf[disp_cols].head(40)
        data = _df_to_table_data(sub, max_rows=40)
        nc = max(len(data[0]), 1)
        mw = min(6.5 / nc, 1.6)
        mt = Table(data, colWidths=[mw * inch] * nc)
        mt.setStyle(TableStyle(_table_style_modern() + [("FONTSIZE", (0, 0), (-1, -1), 6)]))
        story.append(mt)


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
    Generate a professional PDF QC report with executive summary and optional Cloud vs LinC section.
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
            f"Cloud vs LinC concordance (selected scope): {clin.get('concordance_pct', 0):.2f}% "
            f"({clin.get('n_failed_strict', 0)} strict-field mismatch(es))."
        )
    elif clin and clin.get("error"):
        es_lines.append(f"Cloud vs LinC validation: not included — {clin.get('error')}.")
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
    t = Table(mgmt, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle(_table_style_modern()))
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("QC score detail", h2_style))
    story.append(Paragraph(f"<b>Score: {qc_score}/100</b> — {_escape_xml(qc_status)}", body))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Record-level metrics", h2_style))
    if record_metrics is not None and not record_metrics.empty:
        data = _df_to_table_data(record_metrics)
        tw = Table(data, colWidths=[1.2 * inch] * len(data[0]))
        tw.setStyle(TableStyle(_table_style_modern()))
        story.append(tw)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Header consistency", h2_style))
    story.append(Paragraph(f"Header similarity across files: {header_similarity}%", body))
    if header_table is not None and not header_table.empty:
        data = _df_to_table_data(header_table, max_rows=30)
        col_w = 1.5 * inch if len(data[0]) <= 4 else 1.0 * inch
        th = Table(data, colWidths=[col_w] * len(data[0]))
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
