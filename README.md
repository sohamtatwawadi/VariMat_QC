# VariMAT QC Dashboard

A production-ready web dashboard for uploading and comparing multiple VariMAT files used in genomics variant analysis, with automated QC reports.

## Features

- **Multi-file upload** (2–10 files): `.txt`, `.tsv`, `.txt.gz`
- **Record- and variant-level metrics** using variant key: `CHROM_START_REF_ALT`
- **Duplicate variant analysis** (frequency, top duplicated, histograms)
- **Header consistency** across files and similarity score
- **Variant overlap** (common, unique per file, missing between files)
- **Column value QC** (annotation/classification mismatches)
- **Missing data** analysis and heatmaps
- **QC score** (0–100) with status (Excellent / Good / Fair / Needs attention)
- **Export**: CSV (metrics, variants, mismatches, headers) and **PDF report**
- **Pipeline drift** warning when duplication patterns differ across files

## Requirements

- Python 3.9+
- See `requirements.txt`

## Installation

```bash
cd varimat_qc_tool
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. **Upload** 2–10 VariMAT files (drag & drop or file picker).
2. Wait for automatic analysis (status and record counts appear).
3. Review **KPIs** and **tabs**: Overview, Variant comparison, Duplicate analysis, Header QC, Mismatch analysis, Missing data, QC reports.
4. Use **Search variant** and **Filter by gene** in the QC reports tab.
5. **Download** CSV exports and the PDF report from the QC reports tab.

No coding required.

## Variant identifier

Unique variant key: **CHROM_START_REF_ALT** (e.g. `chr1_12345_A_G`).  
The same variant can appear on multiple rows (transcripts, annotations); the dashboard reports both **record-level** and **unique variant-level** metrics.

## Project structure

```
varimat_qc_tool/
├── app.py              # Streamlit UI and orchestration
├── qc_engine.py        # QC logic (metrics, comparison, duplicates, headers, overlap, mismatches)
├── report_generator.py # PDF and CSV report generation
├── utils.py            # File loading, variant key creation
├── requirements.txt
└── README.md
```

## Troubleshooting: `502` or `AxiosError` on file upload

If **browser upload** fails (often **502 Bad Gateway** or **AxiosError**) for **large VariMAT files** (e.g. 1GB+):

- The app already allows **10GB** per file in `.streamlit/config.toml` and the **Dockerfile**.
- The failure is usually the **host’s reverse proxy** (e.g. **Railway**, nginx) **timing out** or limiting long requests — not Streamlit’s size cap.
- **Fix:** use **☁️ S3 (bucket)** or **📁 Local paths (server disk)** in the dashboard so data is not uploaded through the browser session.
- For Railway, also ensure env vars from `.env.example` (`S3_BUCKET`, AWS keys) if using S3.

## Deployment options

- **Internal server**: Run `streamlit run app.py` on a shared machine and share the URL (use `--server.port` and `--server.address` as needed).
- **Streamlit Community Cloud**: Push the repo and connect to GitHub; deploy the app from the Streamlit Cloud dashboard.
- **AWS / Azure**: Run Streamlit in a container or on a VM; expose via load balancer and optional HTTPS.

## File format

VariMAT files must be tab- or comma-separated with at least:

- `CHROM`
- `START`
- `REF`
- `ALT`

Any number of extra columns (e.g. `GENE_NAME`, `TRANSCRIPT`, `VARCLASS`, `AA_CHANGE`) are supported and used for header comparison and column-level QC.

## Troubleshooting

### "AxiosError: Request failed with status code 403" on file upload

**When running locally:** This is often caused by Streamlit’s XSRF (cross-site request forgery) protection blocking the upload. The project’s `.streamlit/config.toml` already sets `enableXsrfProtection = false` for local use so uploads work. Restart the app after changing config (`streamlit run app.py` from the `varimat_qc_tool` directory).

**When running on a host (e.g. cloud):** A **403 Forbidden** usually means something in front of the app is rejecting the request.

1. **Hosting / proxy body size**
   - Many hosts (e.g. Streamlit Community Cloud, Cloud Run, nginx) enforce a **max request body size** (often 200 MB or 1 GB). Requests over that limit can be rejected with 403 or 413.
   - **Streamlit Community Cloud**: Has its own upload limit; 10 GB may not be supported. Use smaller files or run the app on your own server/VM.
   - **Your own server with nginx**: In the server block, set:
     ```nginx
     client_max_body_size 10240M;
     ```
     and reload nginx.
   - **Cloud Run / other serverless**: Check the platform’s max request size and increase it if possible.

2. **Authentication / IAM**
   - If the app is behind Cloud IAP, Cloud Run auth, or an API gateway, unauthenticated or expired requests get 403. Ensure the browser session is logged in and has permission to hit the upload endpoint.

3. **Run locally to confirm**
   - Run `streamlit run app.py` on your machine and upload the same files. If uploads work locally, the 403 is coming from your deployment (proxy, host, or auth), not from the app code.

## License

Internal use. Adjust as needed for your organization.
