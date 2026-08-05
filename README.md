# mxxsicbeat-backend

Turns a song into a beatmap for [mxxsicbeat](../mxxsicbeat): upload audio, get a JSON beatmap
(per-lane, per-note timings) back once background processing finishes.

## Architecture

- **api** (FastAPI): accepts audio uploads, stores them in MinIO, enqueues a generation job,
  and serves polling/result endpoints. Doesn't run any DSP itself.
- **worker** (BullMQ, via the [`bullmq`](https://pypi.org/project/bullmq/) Python port):
  pulls jobs from Redis, downloads the audio, runs demucs drum-stem separation + kick-onset
  detection (ported from `music-map-extractor`), and writes the resulting beatmap to MongoDB.
- **MongoDB**: source of truth for job status and finished beatmaps.
- **Redis**: BullMQ's transport only - never queried directly by the API.
- **MinIO**: S3-compatible object storage for raw uploaded audio, shared between api (write)
  and worker (read).

```
client -> POST /api/v1/maps -> MinIO (audio) + Mongo (job=queued) + Redis (enqueue)
                                                    |
                                              worker picks up job
                                                    |
                              demucs separate -> kick onset detect -> lane assignment
                                                    |
                                     Mongo (beatmap saved, job=completed)
                                                    |
client -> GET /api/v1/maps/jobs/{id} (poll) -> GET /api/v1/maps/{beatmap_id} (JSON beatmap)
```

## Beatmap JSON shape

```json
{
  "id": "665f1a2b3c4d5e6f7a8b9c0d",
  "title": "song.mp3",
  "lane_count": 2,
  "duration_ms": 183200,
  "bpm": null,
  "notes": [
    { "time_ms": 1234, "lane": 0, "energy": 0.83 },
    { "time_ms": 1580, "lane": 1, "energy": 0.61 }
  ],
  "created_at": "2026-08-03T12:00:00Z"
}
```

Lane assignment is currently a simple round-robin over detected kick onsets (only kick-onset
detection is implemented so far - no pitch/frequency-band signal to place notes more
meaningfully yet). `bpm` and per-note pitch are reserved for a future pass that wires in
`basic-pitch`.

## Running locally

```bash
cp .env.example .env
docker compose up --build
```

This starts Mongo, Redis, MinIO, the API (`:8000`), and the worker.

Generate a map from one of the sample tracks already sitting in `music-map-extractor/`:

```bash
curl -F file=@/Users/kyrylo.shkarun/music-map-extractor/2.mp3 http://localhost:8000/api/v1/maps
# => {"job_id": "...", "status": "queued"}

curl http://localhost:8000/api/v1/maps/jobs/<job_id>
# poll until "status": "completed" (demucs separation takes a while on CPU)

curl http://localhost:8000/api/v1/maps/<beatmap_id>
# => the full beatmap JSON above
```

`lane_count` (form field, default 2, max 8) controls how many lanes notes are spread across:

```bash
curl -F file=@song.mp3 -F lane_count=4 http://localhost:8000/api/v1/maps
```

Health checks: `GET /health` (liveness), `GET /health/ready` (Mongo + Redis reachability).

## Migrations

MongoDB index/schema changes live in [`migrations/`](migrations/README.md) and are applied
explicitly via `uv run python -m migrations.runner` - never at api/worker startup. See that
folder's README for details and how to add one.

## Development

```bash
uv sync --extra worker --group dev   # full env, including demucs/librosa for worker code
uv run ruff check .
uv run pytest
uv run uvicorn app.main:app --reload   # api only, needs mongo/redis/minio reachable
uv run python -m app.domains.maps.jobs.handlers.worker   # worker only, needs the worker extra installed
```

Tests don't require Docker: the API tests override Mongo/storage/queue with in-memory fakes
(`mongomock-motor` + stub storage/queue), and the DSP tests exercise lane assignment against
synthetic onset data rather than real audio.
