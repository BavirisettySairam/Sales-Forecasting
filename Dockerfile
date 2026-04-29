# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps — separate layer, cached until apt packages change
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry + all Python deps before copying source code.
# pip cache mount means torch/prophet/etc are downloaded ONCE and reused
# on every subsequent build — no re-downloading gigabyte packages.
COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --quiet poetry==1.8.3 \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --without dev

# Copy only the source that the API needs (not tests, notebooks, etc.)
COPY src/ ./src/
COPY config/ ./config/

# Non-root user — combined into one layer
RUN groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
