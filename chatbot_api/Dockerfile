FROM python:3.11-slim

WORKDIR /app

# Copy entire backend folder (IMPORTANT)
COPY . .

# fastembed / scientific stacks sometimes need a compiler during pip; purge build deps after install
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir --default-timeout=600 . \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# /app for `src.xxx` imports (uvicorn src.main:app)
# /app/src for bare `ingest.xxx` / `models.xxx` imports inside ingest_pipeline.py
ENV PYTHONPATH=/app:/app/src

EXPOSE 8000

# Run app (main.py is inside src)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]