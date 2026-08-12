from enum import StrEnum

from pydantic import BaseModel


class NoteType(StrEnum):
    DRUM = "drum"
    MELODY = "melody"


class Note(BaseModel):
    time: int
    lane: int
    energy: float
    note_type: NoteType
    duration: int | None = None
    combo: int = 1


class NoteGroup(BaseModel):
    lane_count: int
    notes: list[Note]
