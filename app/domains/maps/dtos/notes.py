from enum import StrEnum

from pydantic import BaseModel


class NoteType(StrEnum):
    DRUM = "drum"
    MELODY = "melody"


class Note(BaseModel):
    """Crosses the BullMQ Redis/JSON boundary as part of kick_handler's and melody_handler's job
    results, so it's a validated pydantic model rather than a plain dict. `duration` (ms) is
    only ever set for melody notes - basic-pitch transcribes a start/end per note, while
    kick-onset detection only ever produces an instant."""

    time: int
    lane: int
    energy: float
    note_type: NoteType
    duration: int | None = None
    combo: int = 1


class NoteGroup(BaseModel):
    """One note type's slice of a beatmap, stored at its own index in `beatmaps.notes` (drum at
    0, melody at 1) so each type can carry its own `lane_count`."""

    lane_count: int
    notes: list[Note]
