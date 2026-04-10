"""
Convert USER_MANUAL.md to USER_MANUAL.pdf using ReportLab.
Run from varimat_qc_tool folder: python md_to_pdf.py
"""

import os
import re
from io import BytesIO

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
    Preformatted,
    PageBreak,
)


def _md_bold_and_code(text: str) -> str:
    """Convert **x** to <b>x</b> and `x` to <font name="Courier" size="8">x</font> for ReportLab."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold: **...** or __...__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Inline code: `...`
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', text)
    return text


def _parse_table(lines: list) -> tuple[list[list[str]], int]:
    """Parse markdown table; return list of rows and number of lines consumed."""
    rows = []
    i = 0
    for i, line in enumerate(lines):
        line = line.rstrip()
        if not line.strip() or not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        # Skip separator row (|---|---|)
        if cells and re.match(r"^[-:]+$", cells[0].replace(" ", "")):
            continue
        rows.append(cells)
    return rows, i + 1


def _table_cell_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Wrap cell text in a Paragraph so it wraps inside the table cell."""
    return Paragraph(_md_bold_and_code(text), style)


def _parse_md(path: str) -> list:
    """Parse markdown into a list of flowables for ReportLab."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="H1", fontSize=18, spaceAfter=12, textColor=colors.HexColor("#1a1a1a"),
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="H2", fontSize=14, spaceAfter=8, spaceBefore=12, textColor=colors.HexColor("#2c5282"),
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="H3", fontSize=11, spaceAfter=6, spaceBefore=8, textColor=colors.HexColor("#2d3748"),
        fontName="Helvetica-Bold",
    ))
    code_style = ParagraphStyle(
        name="Code", fontName="Courier", fontSize=8, leftIndent=20, rightIndent=20,
        backColor=colors.HexColor("#f7fafc"), spaceAfter=8, spaceBefore=4,
    )
    # Table cell style: small font so content wraps inside cell (no overflow)
    table_cell_style = ParagraphStyle(
        name="TableCell", fontName="Helvetica", fontSize=7, leading=8,
        leftIndent=2, rightIndent=2, spaceBefore=0, spaceAfter=0,
    )
    table_header_style = ParagraphStyle(
        name="TableHeader", fontName="Helvetica-Bold", fontSize=7, leading=8,
        leftIndent=2, rightIndent=2, spaceBefore=0, spaceAfter=0,
        textColor=colors.whitesmoke,
    )
    body = styles["Normal"]
    body.spaceAfter = 6

    flowables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            flowables.append(Spacer(1, 6))
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", stripped):
            flowables.append(Spacer(1, 0.2 * inch))
            i += 1
            continue

        # Headers
        if line.startswith("# "):
            flowables.append(Paragraph(_md_bold_and_code(line[2:].strip()), styles["H1"]))
            i += 1
            continue
        if line.startswith("## "):
            flowables.append(Paragraph(_md_bold_and_code(line[3:].strip()), styles["H2"]))
            i += 1
            continue
        if line.startswith("### "):
            flowables.append(Paragraph(_md_bold_and_code(line[4:].strip()), styles["H3"]))
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            if i < len(lines):
                i += 1
            code_text = "\n".join(code_lines)
            flowables.append(Preformatted(code_text, code_style))
            continue

        # Table
        if line.strip().startswith("|"):
            table_rows, consumed = _parse_table(lines[i:])
            if table_rows:
                ncols = len(table_rows[0])
                # Use Paragraphs so cell text wraps; header row uses bold style
                wrapped_rows = []
                for r_idx, row in enumerate(table_rows):
                    cell_style = table_header_style if r_idx == 0 else table_cell_style
                    wrapped_rows.append([
                        _table_cell_paragraph(cell, cell_style) for cell in row
                    ])
                # Column widths so content wraps inside cells (no overflow)
                if ncols == 2:
                    col_widths = [1.5 * inch, 5.0 * inch]
                elif ncols == 3:
                    col_widths = [1.2 * inch, 1.3 * inch, 4.0 * inch]  # e.g. Score | Status | Meaning
                else:
                    col_widths = [(6.5 * inch) / ncols] * ncols
                t = Table(wrapped_rows, colWidths=col_widths)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                flowables.append(t)
                flowables.append(Spacer(1, 8))
            i += consumed
            continue

        # Bullet or numbered list item
        if stripped.startswith("- ") or stripped.startswith("* "):
            text = _md_bold_and_code(stripped[2:])
            flowables.append(Paragraph(f"• {text}", body))
            i += 1
            continue
        if re.match(r"^\d+\.\s", stripped):
            text = _md_bold_and_code(re.sub(r"^\d+\.\s+", "", stripped))
            flowables.append(Paragraph(text, body))
            i += 1
            continue

        # Paragraph (possibly multi-line)
        para_lines = [line.rstrip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") and not lines[i].strip().startswith("|") and not lines[i].strip().startswith("-") and not lines[i].strip().startswith("```") and not re.match(r"^\d+\.\s", lines[i].strip()):
            para_lines.append(lines[i].rstrip())
            i += 1
        text = " ".join(para_lines)
        if text.strip():
            flowables.append(Paragraph(_md_bold_and_code(text), body))

    return flowables


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(script_dir, "USER_MANUAL.md")
    pdf_path = os.path.join(script_dir, "USER_MANUAL.pdf")

    if not os.path.isfile(md_path):
        print(f"Not found: {md_path}")
        return 1

    flowables = _parse_md(md_path)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(flowables)
    print(f"Created: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
