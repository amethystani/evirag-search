FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# System deps for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-download SPECTER model at build time so queries don't wait for it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('allenai-specter')"

# /data is where the index files live (downloaded at runtime from HF Datasets)
RUN mkdir -p /data && chown -R user:user /data /app

USER user
ENV HF_HOME=/data/.hf_cache
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
