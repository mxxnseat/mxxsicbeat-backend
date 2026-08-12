from typing import Any


def make_note(**overrides: Any) -> dict:
    """A single note as it's stored in Mongo / returned over the API - drum-shaped by default
    (no duration, combo 1); pass note_type="melody" + duration to get a melody-shaped one."""
    note = {
        "time": 1000,
        "lane": 0,
        "energy": 0.5,
        "duration": None,
        "note_type": "drum",
        "combo": 1,
    }
    note.update(overrides)
    return note


def make_note_group(*, notes: list[dict] | None = None, **overrides: Any) -> dict:
    group = {
        "lane_count": 2,
        "notes": notes if notes is not None else [],
    }
    group.update(overrides)
    return group
