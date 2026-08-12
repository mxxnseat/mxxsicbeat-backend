from datetime import UTC, datetime
from typing import Any

from tests.fixtures.notes import make_note_group


def make_beatmap_doc(*, notes: list[dict] | None = None, **overrides: Any) -> dict:
    beatmap = {
        "job_id": "job-1",
        "title": "song.mp3",
        "lane_count": 2,
        "duration": 180000,
        "bpm": None,
        "notes": notes if notes is not None else [make_note_group(), make_note_group()],
        "created_at": datetime.now(UTC),
    }
    beatmap.update(overrides)
    return beatmap
