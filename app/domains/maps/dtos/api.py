from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.domains.maps.dtos.notes import NoteType

JobStatus = Literal["queued", "processing", "completed", "failed"]


class BeatmapNote(BaseModel):
    time: int
    lane: int
    energy: float
    duration: int | None
    note_type: NoteType


class BeatmapNoteGroup(BaseModel):
    lane_count: int
    notes: list[BeatmapNote]


class Beatmap(BaseModel):
    id: str
    title: str
    duration: int
    bpm: float | None = None
    notes: tuple[BeatmapNoteGroup, BeatmapNoteGroup]
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
