FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# /tmp is writable on Railway; parquet sidecars and downloads go there
ENV TMPDIR=/tmp

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=10240", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
