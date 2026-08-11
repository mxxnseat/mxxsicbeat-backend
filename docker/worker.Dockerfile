FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /srv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra worker

COPY app ./app

RUN uv sync --frozen --no-dev --extra worker

FROM python:3.11-slim

# ffmpeg is required by librosa/soundfile/demucs for decoding compressed audio (mp3, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY --from=builder /srv/.venv ./.venv
COPY --from=builder /srv/app ./app

ENV PATH="/srv/.venv/bin:$PATH"

# This image is shared by all three job handler processes (kick_handler, midi_handler,
# beatmap_handler) - the module to run is supplied as the container's command, e.g.:
#   python -m app.domains.maps.jobs.handlers.kick_handler
CMD ["python", "-m", "app.domains.maps.jobs.handlers.kick_handler"]
