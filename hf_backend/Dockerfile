FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY ollama_cloud_client.py .

RUN chown -R user:user /app
USER user

# HF Spaces expects port 7860 for the UI, but our API runs on 8000.
# We expose both and use 7860 as the primary (HF proxies it).
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
