import numpy as np
import pytest
from bson import ObjectId
from pydantic import ValidationError

from app.domains.maps.dsp.kick_onseter import KickOnseter
from app.domains.maps.dsp.stem_separator import SeparatedStems
from app.domains.maps.dtos.internal import NewBeatmap
from app.domains.maps.dtos.notes import Note, NoteGroup, NoteType
from app.domains.maps.jobs.handlers import kick_handler as kick_handler_module
from app.domains.maps.jobs.handlers import melody_handler as melody_handler_module
from app.domains.maps.jobs.handlers import stem_handler as stem_handler_module
from app.domains.maps.jobs.handlers.beatmap_handler import BeatmapHandler
from app.domains.maps.jobs.handlers.kick_handler import KickHandler
from app.domains.maps.jobs.handlers.melody_handler import MelodyHandler
from app.domains.maps.jobs.handlers.stem_handler import StemHandler
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService
from app.domains.maps.services.notes_service import calculate_melody_combo


class FakeJob:
    def __init__(self, data: dict, children: dict | None = None) -> None:
        self.data = data
        self._children = children or {}

    async def getChildrenValues(self) -> dict:
        return self._children


@pytest.fixture
def job_service(fake_db) -> MapJobService:
    return MapJobService(MapRepository(fake_db))


async def _insert_job(fake_db, **overrides) -> str:
    doc = {
        "status": "queued",
        "original_filename": "song.mp3",
        "object_key": "audio/x/song.mp3",
        "lane_count": 2,
        "beatmap_id": None,
        "error": None,
    }
    doc.update(overrides)
    result = await fake_db.map_jobs.insert_one(doc)
    return str(result.inserted_id)


async def _get_job(fake_db, job_id: str) -> dict:
    return await fake_db.map_jobs.find_one({"_id": ObjectId(job_id)})


async def test_stem_handler_separates_uploads_and_enqueues_next_flow(
    fake_db, fake_storage, fake_flow_producer, job_service, monkeypatch, tmp_path
):
    job_id = await _insert_job(fake_db)
    fake_storage.objects["audio/x/song.mp3"] = b"fake-audio-bytes"

    drum_stem_path = tmp_path / "drums.wav"
    drum_stem_path.write_bytes(b"fake-drum-bytes")
    melody_stem_path = tmp_path / "no_vocals.wav"
    melody_stem_path.write_bytes(b"fake-melody-bytes")

    canned_stems = SeparatedStems(drum_stem_path=drum_stem_path, melody_stem_path=melody_stem_path)
    monkeypatch.setattr(stem_handler_module, "separate_stems", lambda *args, **kwargs: canned_stems)
    monkeypatch.setattr(stem_handler_module, "track_duration", lambda *args, **kwargs: 5000)
    monkeypatch.setattr(stem_handler_module, "detect_bpm", lambda *args, **kwargs: 120.0)

    handler = StemHandler(job_service, fake_storage, fake_flow_producer)
    job = FakeJob(
        {"job_id": job_id, "object_key": "audio/x/song.mp3", "original_filename": "song.mp3", "lane_count": 2}
    )

    result = await handler(job, "token")

    assert result == {"job_id": job_id}
    assert (await _get_job(fake_db, job_id))["status"] == "processing"
    assert fake_storage.objects[fake_storage.drum_key(job_id)] == b"fake-drum-bytes"
    assert fake_storage.objects[fake_storage.melody_key(job_id)] == b"fake-melody-bytes"

    assert len(fake_flow_producer.added_flows) == 1
    flow = fake_flow_producer.added_flows[0]
    assert flow["data"]["job_id"] == job_id
    assert flow["data"]["duration"] == 5000
    assert flow["data"]["bpm"] == 120.0

    children = flow["children"]
    assert {child["queueName"] for child in children} == {"kick-onset-detection", "melody-extraction"}
    melody_child = next(child for child in children if child["queueName"] == "melody-extraction")
    assert melody_child["data"]["bpm"] == 120.0


async def test_stem_handler_marks_failed_and_reraises_on_error(
    fake_db, fake_storage, fake_flow_producer, job_service
):
    job_id = await _insert_job(fake_db)
    # object_key not present in fake_storage.objects -> download raises KeyError

    handler = StemHandler(job_service, fake_storage, fake_flow_producer)
    job = FakeJob(
        {"job_id": job_id, "object_key": "missing/key", "original_filename": "song.mp3", "lane_count": 2}
    )

    with pytest.raises(KeyError):
        await handler(job, "token")

    updated = await _get_job(fake_db, job_id)
    assert updated["status"] == "failed"
    assert updated["error"]


async def test_kick_handler_downloads_drum_stem_and_returns_notes(
    fake_db, fake_storage, job_service, monkeypatch, note_factory
):
    job_id = await _insert_job(fake_db)
    fake_storage.objects[fake_storage.drum_key(job_id)] = b"fake-drum-bytes"
    fake_storage.objects[fake_storage.original_key(job_id, "song.mp3")] = b"fake-original-bytes"

    onseter = KickOnseter()
    monkeypatch.setattr(kick_handler_module, "has_significant_drum_energy", lambda drums, original: True)
    monkeypatch.setattr(kick_handler_module, "detect_kick_onsets", lambda path: ([0.0, 0.5], onseter))

    handler = KickHandler(job_service, fake_storage)
    job = FakeJob(
        {
            "job_id": job_id,
            "lane_count": 2,
            "object_key": "audio/x/song.mp3",
            "original_filename": "song.mp3",
        }
    )

    result = await handler(job, "token")

    assert result == {
        "notes": [
            note_factory(time=0, lane=0, energy=0.0),
            note_factory(time=500, lane=1, energy=0.0),
        ]
    }


async def test_kick_handler_returns_no_notes_when_drum_stem_lacks_energy(
    fake_db, fake_storage, job_service, monkeypatch
):
    """Instrumental-only tracks still get a "drums" stem out of demucs - it's a learned mask,
    not a drum-presence classifier - but it's just separation leakage (bass bleed, artifacts)
    rather than real hits. kick_handler should skip onset detection entirely rather than let
    librosa's relative-threshold onset_detect flag that leakage as kicks."""
    job_id = await _insert_job(fake_db)
    fake_storage.objects[fake_storage.drum_key(job_id)] = b"fake-drum-bytes"
    fake_storage.objects[fake_storage.original_key(job_id, "song.mp3")] = b"fake-original-bytes"

    monkeypatch.setattr(kick_handler_module, "has_significant_drum_energy", lambda drums, original: False)
    monkeypatch.setattr(
        kick_handler_module,
        "detect_kick_onsets",
        lambda path: pytest.fail("detect_kick_onsets should be skipped when drum energy is insignificant"),
    )

    handler = KickHandler(job_service, fake_storage)
    job = FakeJob(
        {
            "job_id": job_id,
            "lane_count": 2,
            "object_key": "audio/x/song.mp3",
            "original_filename": "song.mp3",
        }
    )

    result = await handler(job, "token")

    assert result == {"notes": []}


async def test_kick_handler_marks_failed_and_reraises_on_error(fake_db, fake_storage, job_service):
    job_id = await _insert_job(fake_db)
    # drum stem not present in fake_storage.objects -> download raises KeyError

    handler = KickHandler(job_service, fake_storage)
    job = FakeJob(
        {
            "job_id": job_id,
            "lane_count": 2,
            "object_key": "audio/x/song.mp3",
            "original_filename": "song.mp3",
        }
    )

    with pytest.raises(KeyError):
        await handler(job, "token")

    updated = await _get_job(fake_db, job_id)
    assert updated["status"] == "failed"
    assert updated["error"]


async def test_melody_handler_marks_failed_on_invalid_payload(fake_db, fake_storage, job_service):
    job_id = await _insert_job(fake_db)
    handler = MelodyHandler(job_service, fake_storage)
    job = FakeJob({"job_id": job_id})  # missing required fields (lane_count, bpm)

    with pytest.raises(ValidationError):
        await handler(job, "token")

    assert (await _get_job(fake_db, job_id))["status"] == "failed"


async def test_melody_handler_downloads_melody_stem_and_returns_notes(
    fake_db, fake_storage, job_service, monkeypatch, note_factory
):
    job_id = await _insert_job(fake_db)
    fake_storage.objects[fake_storage.melody_key(job_id)] = b"fake-melody-bytes"
    note_combo = calculate_melody_combo(500, 120)
    canned_note = Note(time=0, lane=0, energy=0.8, note_type=NoteType.MELODY, duration=500, combo=note_combo)
    monkeypatch.setattr(melody_handler_module.librosa, "load", lambda *args, **kwargs: (np.zeros(100), 22050))
    monkeypatch.setattr(melody_handler_module, "build_melody_notes", lambda y, sr, bpm, lane_count: [canned_note])

    handler = MelodyHandler(job_service, fake_storage)
    job = FakeJob({"job_id": job_id, "lane_count": 2, "bpm": 120.0})

    result = await handler(job, "token")

    assert result == {
        "notes": [
            note_factory(time=0, lane=0, energy=0.8, note_type="melody", duration=500, combo=note_combo),
        ]
    }


async def test_beatmap_handler_merges_kick_child_and_finalizes(
    fake_db, job_service, note_factory, note_group_factory
):
    job_id = await _insert_job(fake_db)
    handler = BeatmapHandler(job_service)
    job = FakeJob(
        {
            "job_id": job_id,
            "object_key": "audio/x/song.mp3",
            "original_filename": "song.mp3",
            "lane_count": 2,
            "duration": 5000,
            "bpm": 120.0,
        },
        children={
            "bull:kick-onset-detection:abc": {
                "notes": [{"time": 100, "lane": 0, "energy": 0.9, "note_type": "drum", "duration": None}],
            },
            "bull:melody-extraction:def": {"notes": []},
        },
    )

    result = await handler(job, "token")

    updated = await _get_job(fake_db, job_id)
    assert updated["status"] == "completed"
    assert updated["beatmap_id"] == result["beatmap_id"]

    beatmap = await fake_db.beatmaps.find_one({"_id": ObjectId(result["beatmap_id"])})
    assert beatmap["title"] == "song.mp3"
    assert beatmap["object_key"] == "audio/x/song.mp3"
    assert beatmap["duration"] == 5000
    assert beatmap["bpm"] == 120.0
    assert beatmap["notes"] == [
        note_group_factory(notes=[note_factory(time=100, lane=0, energy=0.9)]),
        note_group_factory(),
    ]


async def test_beatmap_handler_marks_failed_when_kick_child_missing(fake_db, job_service):
    job_id = await _insert_job(fake_db)
    handler = BeatmapHandler(job_service)
    job = FakeJob(
        {
            "job_id": job_id,
            "object_key": "audio/x/song.mp3",
            "original_filename": "song.mp3",
            "lane_count": 2,
            "duration": 5000,
            "bpm": 120.0,
        },
        children={},
    )

    with pytest.raises(KeyError):
        await handler(job, "token")

    assert (await _get_job(fake_db, job_id))["status"] == "failed"


async def test_map_job_service_finalize_beatmap(fake_db, job_service):
    job_id = await _insert_job(fake_db)
    beatmap = NewBeatmap(
        job_id=job_id,
        object_key="audio/x/song.mp3",
        title="song.mp3",
        duration=1000,
        bpm=None,
        notes=(NoteGroup(lane_count=2, notes=[]), NoteGroup(lane_count=2, notes=[])),
    )

    beatmap_id = await job_service.finalize_beatmap(beatmap)

    updated = await _get_job(fake_db, job_id)
    assert updated["status"] == "completed"
    assert updated["beatmap_id"] == beatmap_id


async def test_map_job_service_mark_processing_and_failed(fake_db, job_service):
    job_id = await _insert_job(fake_db)

    await job_service.mark_processing(job_id)
    assert (await _get_job(fake_db, job_id))["status"] == "processing"

    await job_service.mark_failed(job_id, "boom")
    updated = await _get_job(fake_db, job_id)
    assert updated["status"] == "failed"
    assert updated["error"] == "boom"
