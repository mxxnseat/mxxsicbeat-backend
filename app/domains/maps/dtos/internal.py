from dataclasses import dataclass


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
    lane_count: int
    duration_ms: int
    bpm: float | None
    notes: list[dict]
