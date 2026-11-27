# Stage 1: Build React + TypeScript + Tailwind SPA (zero Node in final image)
FROM node:22-alpine AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Lightweight Python 3.12 + uv runtime (ONNX fastembed, no PyTorch)
FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    DATA_DIR=/app/data

WORKDIR /app
COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY src/ ./src/
COPY seeds/ ./seeds/
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

RUN uv sync --frozen --no-dev && \
    mkdir -p /app/data/raw /app/data/processed /app/data/index && \
    useradd --create-home --shell /bin/sh astropilote && \
    chown -R astropilote:astropilote /app

USER astropilote
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD uv run --no-sync python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uv run --no-sync alembic upgrade head && exec uv run --no-sync uvicorn astropilote.app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
