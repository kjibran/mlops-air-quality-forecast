FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["sh", "-c", "uvicorn mlops_air_quality_forecast.api:app --host 0.0.0.0 --port ${PORT:-8000}"]