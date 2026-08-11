FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /srv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app

RUN uv sync --frozen --no-dev

FROM python:3.11-slim

WORKDIR /srv

COPY --from=builder /srv/.venv ./.venv
COPY --from=builder /srv/app ./app

ENV PATH="/srv/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
