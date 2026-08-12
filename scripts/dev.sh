#!/usr/bin/env bash
# Runs the api and worker locally (via uv) against dockerized mongo/redis/minio, so both
# processes get uvicorn's --reload / fast Python restarts instead of a docker image rebuild.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example"
fi

echo "starting mongo, redis, minio..."
docker compose up -d --wait mongo redis minio cdn

# worker needs the "worker" extra (demucs/librosa/etc.) - make sure it's installed, since a
# plain `uv run` only syncs the base dependency set
uv sync --extra worker --group dev

# .env's mongo/redis/minio hosts are the docker-compose *service names*, only resolvable from
# inside the compose network. Running api/worker on the host instead, they need localhost -
# exported vars take priority over .env in pydantic-settings, so this overrides just these three.
export MONGODB_URI="mongodb://localhost:27017"
export REDIS_URL="redis://localhost:6379"
export MINIO_ENDPOINT_URL="http://localhost:9000"
export MAPS_STORAGE_PUBLIC_BASE_URL="http://localhost:8080"
export MAX_UPLOAD_SIZE_MB=100

echo "running migrations..."
uv run python -m migrations.runner

echo "Check types..."
uv run pyright

# kill both background jobs together on Ctrl+C or script exit
trap 'kill 0' EXIT

uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
uv run python -m app.domains.maps.jobs.handlers.stem_handler &
uv run python -m app.domains.maps.jobs.handlers.kick_handler &
uv run python -m app.domains.maps.jobs.handlers.melody_handler &
uv run python -m app.domains.maps.jobs.handlers.beatmap_handler &

wait
