from datetime import datetime
from typing import Literal

from pydantic import BaseModel

JobStatus = Literal["queued", "processing", "completed", "failed"]


class BeatmapNote(BaseModel):
    time_ms: int
    lane: int
    energy: float


class Beatmap(BaseModel):
    id: str
    title: str
    lane_count: int
    duration_ms: int
    bpm: float | None = None
    notes: list[BeatmapNote]
    created_at: datetime
    audio_url: str | None = None


class MapJobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class MapJobOut(BaseModel):
    job_id: str
    status: JobStatus
    beatmap_id: str | None = None
    error: str | None = None
