from dataclasses import dataclass

from app.domains.maps.dtos.notes import NoteGroup


@dataclass(frozen=True, slots=True)
class NewMapJob:
    original_filename: str
    object_key: str | None
    lane_count: int


@dataclass(frozen=True, slots=True)
class NewBeatmap:
    job_id: str
    object_key: str
    title: str
    duration: int
    bpm: float | None
    notes: tuple[NoteGroup, NoteGroup]
