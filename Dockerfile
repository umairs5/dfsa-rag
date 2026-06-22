FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker layer caching: if requirements.txt
# hasn't changed, Docker reuses this layer instead of reinstalling)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY retrieval/ retrieval/
COPY generation/ generation/
COPY eval/ eval/
COPY api/ api/

# Copy data files (small — chunks.jsonl ~100KB, bm25_index ~80KB)
# Qdrant data lives in the Qdrant container, not here
COPY data/processed/chunks.jsonl data/processed/chunks.jsonl
COPY data/bm25_index.pkl data/bm25_index.pkl

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
