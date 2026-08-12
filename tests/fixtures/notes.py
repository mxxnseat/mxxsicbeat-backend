from typing import Any


def make_note(**overrides: Any) -> dict:
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
