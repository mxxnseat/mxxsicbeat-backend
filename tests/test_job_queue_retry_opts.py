from app.domains.maps.jobs.queues.queue import build_generate_beatmap_flow, build_separate_stems_job


def test_separate_stems_job_carries_retry_opts():
    job = build_separate_stems_job(
        job_id="job-1", object_key="job-1/original/song.mp3", original_filename="song.mp3", lane_count=2
    )

    assert job["opts"]["attempts"] >= 1
    assert "backoff" in job["opts"]


def test_generate_beatmap_flow_carries_retry_opts_on_root_and_children():
    flow = build_generate_beatmap_flow(
        job_id="job-1",
        object_key="job-1/original/song.mp3",
        original_filename="song.mp3",
        lane_count=2,
        duration=5000,
        bpm=120,
    )

    assert "attempts" in flow["opts"]
    assert "backoff" in flow["opts"]

    for child in flow["children"]:
        assert child["opts"]["failParentOnFailure"] is True
        assert "attempts" in child["opts"]
        assert "backoff" in child["opts"]
