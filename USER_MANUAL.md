# VariMAT QC Tool — User Manual

This manual explains how to use the VariMAT QC Tool to compare and quality-check multiple VariMAT variant files. Use it when you have 2–10 files from the same or different pipelines and want to see how they agree, where they differ, and get a single QC score and downloadable reports.

---

## 1. What This Tool Does

The VariMAT QC Tool:

- **Loads** 2–10 VariMAT files (tab-separated or gzipped text with columns such as CHROM, START, REF, ALT and annotation columns).
- **Compares** them using a unique variant key: **CHROM_START_REF_ALT** (e.g. `chr1_12345_A_G`).
- **Computes** record-level and variant-level metrics, duplicate patterns, header consistency, variant overlap, and annotation mismatches.
- **Produces** a **QC score (0–100)** and status (Excellent / Good / Fair / Needs attention).
- **Lets you download** CSV exports and a PDF report for sharing or archiving.

You can run it in two ways:

1. **Web dashboard** (Streamlit) — upload or point to files in a browser and click through tabs.
2. **Jupyter notebook** — run the same QC from a notebook and optionally save reports to a folder.

---

## 2. Folder Contents

After copying the tool to a folder (e.g. on a drive), you should see:

| Item | Description |
|------|-------------|
| `app.py` | Streamlit web app (dashboard). |
| `qc_engine.py` | Core QC logic (metrics, overlap, mismatches, etc.). |
| `report_generator.py` | Builds CSV and PDF reports. |
| `utils.py` | File loading and variant key creation. |
| `run_qc_notebook.py` | Script for running QC from Python/Jupyter or command line. |
| `VariMAT_QC_Run.ipynb` | Jupyter notebook with runnable cells. |
| `requirements.txt` | Python package list. |
| `.streamlit/config.toml` | Streamlit settings (upload size, theme). |
| `USER_MANUAL.md` | This manual (source). |
| `USER_MANUAL.pdf` | PDF version of this manual (for printing/sharing). |
| `md_to_pdf.py` | Script to regenerate the PDF from USER_MANUAL.md: `python md_to_pdf.py` |
| `clinical_concordance.py` | Cloud vs LinC transcript-level cell concordance engine. |
| `tests/test_clinical_concordance.py` | Unit tests for concordance logic. |

---

## 3. Requirements and Installation

- **Python 3.9 or newer**
- **Dependencies:** Install once in the folder where the tool lives:

```bash
cd <folder_containing_varimat_qc_tool>
pip install -r requirements.txt
```

Main packages: `streamlit`, `pandas`, `pyarrow`, `plotly`, `reportlab`, and a few others (see `requirements.txt`).

---

## 4. Input File Format

- **Formats:** `.txt`, `.tsv`, or `.txt.gz` (gzipped).
- **Required columns (exact names):**
  - `CHROM` — chromosome
  - `START` — position
  - `REF` — reference allele
  - `ALT` — alternate allele

Any extra columns (e.g. gene, transcript, classification, amino acid change) are used for header comparison and for finding annotation mismatches. Column names are trimmed of leading/trailing spaces.

- **Limits:** 2–10 files per run; up to 10 GB per file in the dashboard (configurable in `.streamlit/config.toml`).

---

## 5. How to Use — Web Dashboard

### 5.1 Start the dashboard

From a terminal, in the folder that contains `app.py`:

```bash
streamlit run app.py
```

Then open the URL shown (e.g. `http://localhost:8501`) in your browser.

### 5.2 Load your files

You can load files in two ways:

**Option A — Load from local paths (recommended for large files)**

1. Expand **“Load from local paths”**.
2. Enter one file path per line (full paths, e.g. `C:\Data\file1.txt` or `/home/user/file1.txt`).
3. Click **“Load from paths”**.
4. Wait until the **Upload status** table shows all files with **✔ OK** and record counts.

**Option B — Browse and upload**

1. Use **“Browse files”** to select 2–10 VariMAT files.
2. After selection, the app will load and parse them; the **Upload status** table will show progress and **✔ OK** when done.

### 5.3 Run QC and read results

- Once at least 2 files show **✔ OK** in the upload table, the **Full QC** runs automatically (or you can trigger it if the UI has a run button).
- A **Key metrics** section and **Management summary** appear at the top.
- Use the **tabs** to explore: Overview, Variant comparison, Duplicate analysis, Header QC, Mismatch analysis, Missing data, QC reports.
- In **QC reports** you can:
  - Preview data, **Search variant** by key (e.g. `chr1_12345_A_G`), **Filter by gene**.
  - **Download** individual CSV files and the **PDF report**.

Details of each section are in **Section 7 (Understanding the results)**.

---

## 6. How to Use — Jupyter Notebook

### 6.1 Open the notebook

1. Start Jupyter (e.g. `jupyter notebook` or JupyterLab) and go to the folder that contains the VariMAT QC files.
2. Open **`VariMAT_QC_Run.ipynb`**.

### 6.2 Run the cells in order

1. **Cell 1 (Setup and import)**  
   Run this first. It adds the correct folder to Python’s path and imports `run_qc`. You should see a message like “Import OK.”

2. **Cell 2 (Set file paths)**  
   Edit the `paths` list to your actual VariMAT file paths. Set `output_dir` to a folder where you want CSV and PDF reports (e.g. `"./qc_output"`), or `None` to skip writing files.

3. **Cell 3 (Run QC)**  
   Runs the full QC, prints a short summary in the notebook, and, if `output_dir` is set, writes all CSV and PDF reports into that folder.

4. **Cell 4 (Optional)**  
   Inspect `results` (e.g. `results["qc_score"]`, `results["dataframes"]`) for further analysis.

Use **Run → Run All Cells** to run everything in order, or run cell by cell with **Shift+Enter**.

### 6.3 Command-line alternative (no Jupyter)

From a terminal, in the folder that contains `run_qc_notebook.py`:

```bash
python run_qc_notebook.py "/path/to/file1.txt" "/path/to/file2.txt"
```

Reports are not written unless you set `OUTPUT_DIR` inside the script or pass paths and output dir from your own script.

---

## 7. Understanding the Results

### 7.1 Key metrics (top of results)

| Metric | Meaning |
|--------|--------|
| **Total files** | Number of VariMAT files compared (2–10). |
| **Total records** | Sum of all data rows across files (one variant can have multiple rows, e.g. transcripts). |
| **Unique variants (union)** | Number of distinct variant keys (CHROM_START_REF_ALT) across all files. |
| **Duplicate variant entries** | Extra rows that share the same variant key (Total records − unique variants, summed across files). |
| **Variants common in all files** | Variant keys that appear in every file. |
| **Variants unique to one file** | Variant keys that appear in exactly one file. |
| **Header consistency score** | Percentage of column overlap across files (100% = identical column sets). |
| **QC Score** | Overall score 0–100 and status (see below). |

### 7.2 Management summary

Short text summary: how many files, total variants, common variants, variant consistency level, and QC score. Useful for reports or quick checks.

### 7.3 QC score and status

The **QC score** is from 0 to 100. The **status** is:

| Score | Status | Meaning |
|-------|--------|---------|
| 90–100 | **Excellent** | Headers align, most variants shared, few mismatches or missing data. |
| 75–89 | **Good** | Minor differences in overlap or annotations. |
| 50–74 | **Fair** | Notable differences (e.g. many mismatches or low overlap). |
| 0–49 | **Needs attention** | Large differences; worth investigating pipeline versions or inputs. |

The score is reduced for: header differences, low fraction of common variants, many annotation mismatches, many inconsistent annotations, many high-NA columns, and differing duplicate patterns between files (pipeline drift).

### 7.4 Overview tab

- **Record-level summary table:** Per file: total rows, unique variants, duplicate rows, size (MB).
- **Unique variant summary table:** Per file: unique variant count and “only in this file” count.
- **QC score gauge:** Visual 0–100 score.

### 7.5 Variant comparison tab

- **Variant overlap heatmap:** Rows and columns = file names; cell = number of variant keys shared by that pair (or same file = total in that file). Helps see which files agree most.
- **Common variants (in all files):** Count and a table of up to 500 variant IDs present in every file.
- **Unique per file:** For each file, an expandable list of variant IDs that appear only in that file (up to 500 shown).

### 7.6 Duplicate analysis tab

For each file:

- **Variants with duplicate rows** — number of variant keys that have more than one row (e.g. multiple transcripts).
- **Total duplicate entries** — total extra rows beyond one per variant.
- **Top duplicated** — variant keys with the most rows.
- **Histogram** — how many variants have 1, 2, 3, … rows.

If duplication patterns differ a lot between files, the tool may show a **pipeline drift** warning (e.g. different pipeline versions).

### 7.7 Header QC tab

- **Header similarity:** Percentage (0–100%) of columns that are shared across files.
- **Column table:** Each row is a column name; each column is a file with “Yes”/“No” for presence. Use this to see which columns exist only in some files.

### 7.8 Mismatch analysis tab

- **Column value mismatches:** For variants present in more than one file, values in **shared columns** are compared. Each row is one variant and one column where at least two files disagree (with file-specific values). First 500 rows shown; full set can be in the CSV export.
- **Variant consistency (grouped):** Variants where at least one annotation column has different values across files (e.g. different transcript or classification). First 300 rows shown.

### 7.9 Missing data tab

- **Heatmap:** Rows = data columns, columns = files; color = % of missing (NA) values. Red = more missing.
- **Empty columns:** Columns that are 100% NA in all files.
- **High NA columns:** Columns with ≥50% NA in any file.

### 7.10 QC reports tab

- **Data preview:** Select a file and see the first 100 rows.
- **Search variant:** Enter a variant key (e.g. `chr1_12345_A_G`) to see in which files it appears and its rows.
- **Filter by gene:** If a gene column exists (e.g. GENE_NAME, GENE, Gene), pick a gene and see per-file row counts and a sample of rows.
- **Download reports:** Buttons to download each CSV and the PDF report (see Section 8).

---

## 8. Exported Reports (CSV and PDF)

### 8.1 When exports are created

- **Dashboard:** Use the **Download** buttons in the **QC reports** tab.
- **Notebook / script:** Exports are written only if you set `output_dir` (e.g. `"./qc_output"`); all CSV files and one PDF are saved there.

### 8.2 CSV files (what each contains)

| File name | Content |
|-----------|---------|
| `qc_summary.csv` | One row: files_compared, total_records, unique_variants, common_in_all, qc_score, qc_status, header_similarity_pct. |
| `record_metrics.csv` | One row per file: File, Total Rows, Unique Variants, Duplicate Rows, Size (MB). |
| `union_of_all_variants.csv` | One column: all unique variant IDs (union across files). |
| `variants_common_all_files.csv` | One column: variant IDs present in every file. |
| `variants_unique_to_<filename>.csv` | One column: variant IDs that appear only in that file (one CSV per file). |
| `column_mismatches.csv` | Rows: Variant ID, Column, and one column per file with the value in that file (for mismatched columns only). |
| `header_comparison.csv` | One row per data column; columns = files with Yes/No for presence. |
| `missing_in_<B>_but_in_<A>.csv` | Variant IDs that are in file A but not in file B (one CSV per ordered pair when there are missing variants). |

### 8.3 PDF report

Single PDF (e.g. `QC_Report_YYYYMMDD_HHMM.pdf`) formatted for **executive and client review**:

- **Executive summary** (bulleted overview at the top).
- Management summary table, QC score, record-level metrics.
- Header consistency and variant overlap summaries.
- Multi-file column mismatch count and duplicate variant summary.
- **Separate page — Executive validation — Cloud vs LinC** (only if you ran that analysis in the dashboard): plain-language purpose, methodology (join key, optional **all genes** vs one gene, optional **column subset**, on-target filter, strict vs tolerant columns, numeric tolerance with **Cloud as reference**), mismatch export cap (first 250k detail rows; totals still exact), limitations, KPI table, validation summary line, and a short mismatch appendix (full list in CSV up to the cap).

The PDF is not a regulatory sign-off; it documents metrics and findings for leadership and partners.

### 8.4 Executive validation — Cloud vs LinC (dashboard)

After loading **at least two** files, open **Executive validation — Cloud vs LinC** (under the management summary).

1. Choose which file is **Cloud (reference)** and which is **LinC**.
2. Optionally restrict to **on-target** rows using column **`VARIANT_LOCATION`** = `ONTARGET` (values `OFFTARGET` are excluded when enabled).
3. Either check **Compare all genes** (skips the per-gene filter; all rows after other filters are in scope) **or** select **one gene** (must appear in both files after filters).
4. Optionally limit **columns to compare** (multiselect): leave empty to compare every shared annotation column except keys / gene / `VARIANT_LOCATION`. A smaller set runs faster on huge files.
5. Set **strict columns** (default includes `CDNA_CHG`, `AA_CHG`, and HGVS genomic-style columns if present): these must match **exactly** (no numeric tolerance).
6. Set **numeric tolerance** (±%) for other columns: LinC values are compared to **Cloud** as reference.
7. Click **Run Cloud vs LinC validation**. Comparisons are **vectorized** in Python (pandas/numpy). For very large inputs, prefer **local paths**, **one gene** or a **column subset**, and expect on the order of **one to a few minutes** for compute (upload time is separate).
8. Download **PDF** and **CSVs** from **QC reports**: if mismatches exist, `clinical_concordance_mismatches.csv` is included (up to **250,000** mismatch detail rows; concordance counts use the full mismatch total).

**Row matching:** Rows are aligned on `CHROM`, `START`, `REF`, `ALT`, and `ENS_TRANS_ID`. Duplicate keys in one file are collapsed (first row kept); counts are shown in the UI and PDF.

Use this for launch-style parity checks between cloud pipeline output and LinC—either for a **focused gene** or for **all genes** when you accept a heavier run.

---

## 9. Troubleshooting

- **“Need at least 2 valid files”**  
  Provide at least two VariMAT files with required columns (CHROM, START, REF, ALT). Check paths or upload again.

- **“Missing required column: …”**  
  Ensure each file has columns named exactly `CHROM`, `START`, `REF`, `ALT` (after trimming spaces).

- **403 or upload fails in browser**  
  When running locally, the project’s `.streamlit/config.toml` disables XSRF for uploads. If you still get 403, try “Load from local paths” instead of browse/upload, or run on a different host and check proxy/body size limits.

- **Slow loading**  
  For very large files (e.g. 1 GB+), use **Load from local paths** so files are read from disk instead of through the browser. The notebook/script also reads from paths and can be faster.

- **Jupyter: “No module named qc_engine”**  
  Run the first cell of `VariMAT_QC_Run.ipynb` first; it adds the correct folder to the path. If the notebook is not in the same folder as `qc_engine.py`, move the notebook into the same folder as the tool or adjust the path in that cell.

- **Out of memory**  
  Reduce the number or size of files per run, or run on a machine with more RAM.

---

## 10. Quick Reference

| Goal | Action |
|------|--------|
| Run in browser | `streamlit run app.py` → open URL → load files → use tabs and download. |
| Run in Jupyter | Open `VariMAT_QC_Run.ipynb` → run cells 1–3 (set paths in cell 2). |
| Run from terminal | `python run_qc_notebook.py file1.txt file2.txt` (from the tool folder). |
| Get reports on disk | In notebook set `output_dir = "./qc_output"`; in dashboard use Download in QC reports tab. |
| Compare pipelines | Load 2+ files → check Key metrics, Variant comparison, Mismatch analysis, and QC score. |
| Cloud vs LinC validation | Run **Executive validation** → pick Cloud/LinC files → optional **Compare all genes** or one gene → optional column subset → download PDF + `clinical_concordance_mismatches.csv` if needed. |

---

*VariMAT QC Tool — User Manual. For support or customization, refer to the project README and source code.*
