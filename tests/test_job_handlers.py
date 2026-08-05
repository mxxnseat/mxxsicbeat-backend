import pytest
from bson import ObjectId
from pydantic import ValidationError

from app.domains.maps.dsp.pipeline import GeneratedBeatmap
from app.domains.maps.dtos.internal import NewBeatmap
from app.domains.maps.jobs.handlers import kick_handler as kick_handler_module
from app.domains.maps.jobs.handlers.beatmap_handler import BeatmapHandler
from app.domains.maps.jobs.handlers.kick_handler import KickHandler
from app.domains.maps.jobs.handlers.midi_handler import MidiHandler
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService


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


async def test_kick_handler_marks_processing_downloads_and_returns_notes(
    fake_db, fake_storage, job_service, monkeypatch, tmp_path
):
    job_id = await _insert_job(fake_db)
    fake_storage.objects["audio/x/song.mp3"] = b"fake-audio-bytes"

    drum_stem_path = tmp_path / "drums.wav"
    drum_stem_path.write_bytes(b"fake-drum-bytes")
    melody_stem_path = tmp_path / "no_drums.wav"
    melody_stem_path.write_bytes(b"fake-melody-bytes")

    canned = GeneratedBeatmap(
        lane_count=2,
        duration_ms=1000,
        bpm=None,
        notes=[{"time_ms": 0, "lane": 0, "energy": 1.0}],
        drum_stem_path=drum_stem_path,
        melody_stem_path=melody_stem_path,
    )
    monkeypatch.setattr(kick_handler_module, "generate_beatmap", lambda *args, **kwargs: canned)

    handler = KickHandler(job_service, fake_storage)
    job = FakeJob(
        {"job_id": job_id, "object_key": "audio/x/song.mp3", "original_filename": "song.mp3", "lane_count": 2}
    )

    result = await handler(job, "token")

    expected_notes = [{"time_ms": 0, "lane": 0, "energy": 1.0}]
    assert result == {"lane_count": 2, "duration_ms": 1000, "notes": expected_notes}
    assert (await _get_job(fake_db, job_id))["status"] == "processing"
    assert fake_storage.objects[fake_storage.drum_key(job_id)] == b"fake-drum-bytes"
    assert fake_storage.objects[fake_storage.melody_key(job_id)] == b"fake-melody-bytes"


async def test_kick_handler_marks_failed_and_reraises_on_error(fake_db, fake_storage, job_service):
    job_id = await _insert_job(fake_db)
    # object_key not present in fake_storage.objects -> download_to_path raises KeyError

    handler = KickHandler(job_service, fake_storage)
    job = FakeJob(
        {"job_id": job_id, "object_key": "missing/key", "original_filename": "song.mp3", "lane_count": 2}
    )

    with pytest.raises(KeyError):
        await handler(job, "token")

    updated = await _get_job(fake_db, job_id)
    assert updated["status"] == "failed"
    assert updated["error"]


async def test_midi_handler_marks_failed_on_invalid_payload(fake_db, job_service):
    job_id = await _insert_job(fake_db)
    handler = MidiHandler(job_service)
    job = FakeJob({"job_id": job_id})  # missing required fields

    with pytest.raises(ValidationError):
        await handler(job, "token")

    assert (await _get_job(fake_db, job_id))["status"] == "failed"


async def test_beatmap_handler_merges_kick_child_and_finalizes(fake_db, job_service):
    job_id = await _insert_job(fake_db)
    handler = BeatmapHandler(job_service)
    job = FakeJob(
        {
            "job_id": job_id,
            "object_key": "audio/x/song.mp3",
            "original_filename": "song.mp3",
            "lane_count": 2,
        },
        children={
            "bull:kick-onset-detection:abc": {
                "lane_count": 2,
                "duration_ms": 5000,
                "notes": [{"time_ms": 100, "lane": 0, "energy": 0.9}],
            },
            "bull:midi-extraction:def": {"notes": []},
        },
    )

    result = await handler(job, "token")

    updated = await _get_job(fake_db, job_id)
    assert updated["status"] == "completed"
    assert updated["beatmap_id"] == result["beatmap_id"]

    beatmap = await fake_db.beatmaps.find_one({"_id": ObjectId(result["beatmap_id"])})
    assert beatmap["title"] == "song.mp3"
    assert beatmap["object_key"] == "audio/x/song.mp3"
    assert beatmap["lane_count"] == 2
    assert beatmap["duration_ms"] == 5000
    assert beatmap["notes"] == [{"time_ms": 100, "lane": 0, "energy": 0.9}]


async def test_beatmap_handler_marks_failed_when_kick_child_missing(fake_db, job_service):
    job_id = await _insert_job(fake_db)
    handler = BeatmapHandler(job_service)
    job = FakeJob(
        {
            "job_id": job_id,
            "object_key": "audio/x/song.mp3",
            "original_filename": "song.mp3",
            "lane_count": 2,
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
        lane_count=2,
        duration_ms=1000,
        bpm=None,
        notes=[],
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
