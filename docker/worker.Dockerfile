FROM python:3.11-slim

# ffmpeg is required by librosa/soundfile/demucs for decoding compressed audio (mp3, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /srv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra worker

COPY app ./app

RUN uv sync --frozen --no-dev --extra worker

# This image is shared by all three job handler processes (kick_handler, midi_handler,
# beatmap_handler) - the module to run is supplied as the container's command, e.g.:
#   uv run python -m app.domains.maps.jobs.handlers.kick_handler
CMD ["uv", "run", "python", "-m", "app.domains.maps.jobs.handlers.kick_handler"]
