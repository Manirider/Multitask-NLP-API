# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (smaller, better caching)
RUN pip install --user --no-cache-dir --timeout 300 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --user --no-cache-dir --timeout 300 --retries 5 -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

CMD ["sh", "-c", "python src/preprocess.py && python src/train.py && uvicorn src.main:app --host 0.0.0.0 --port 8000"]
