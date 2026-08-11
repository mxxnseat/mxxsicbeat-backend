from datetime import UTC, datetime


async def _upload(
    client,
    *,
    content=b"fake-audio-bytes",
    filename="song.mp3",
    content_type="audio/mpeg",
    lane_count=None,
):
    files = {"file": (filename, content, content_type)}
    data = {"lane_count": str(lane_count)} if lane_count is not None else {}
    return await client.post("/api/v1/maps", files=files, data=data)


async def test_create_map_job_returns_202_and_enqueues(client, fake_flow_producer, fake_storage):
    response = await _upload(client)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert "job_id" in body

    assert len(fake_flow_producer.added_flows) == 1
    flow = fake_flow_producer.added_flows[0]
    assert flow["queueName"] == "stem-separation"
    assert "children" not in flow
    assert flow["data"]["job_id"] == body["job_id"]
    assert flow["data"]["lane_count"] == 2
    assert flow["data"]["object_key"] in fake_storage.objects


async def test_create_map_job_rejects_unsupported_content_type(client):
    response = await _upload(client, content_type="text/plain")

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_audio_type"


async def test_create_map_job_rejects_oversized_file(client, test_config):
    test_config.max_upload_size_mb = 0

    response = await _upload(client)

    assert response.status_code == 413
    assert response.json()["code"] == "audio_too_large"


async def test_create_map_job_rejects_invalid_lane_count(client):
    response = await _upload(client, lane_count=0)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_lane_count"


async def test_create_map_job_respects_custom_lane_count(client, fake_flow_producer):
    response = await _upload(client, lane_count=4)

    assert response.status_code == 202
    assert fake_flow_producer.added_flows[0]["data"]["lane_count"] == 4


async def test_get_job_status_not_found(client):
    response = await client.get("/api/v1/maps/jobs/000000000000000000000000")

    assert response.status_code == 404
    assert response.json()["code"] == "map_job_not_found"


async def test_get_beatmap_not_found(client):
    response = await client.get("/api/v1/maps/000000000000000000000000")

    assert response.status_code == 404
    assert response.json()["code"] == "beatmap_not_found"


async def test_job_status_and_beatmap_round_trip(client, fake_db):
    now = datetime.now(UTC)
    job_result = await fake_db.map_jobs.insert_one(
        {
            "status": "completed",
            "original_filename": "song.mp3",
            "object_key": "audio/x/song.mp3",
            "lane_count": 2,
            "beatmap_id": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    beatmap_result = await fake_db.beatmaps.insert_one(
        {
            "job_id": str(job_result.inserted_id),
            "title": "song.mp3",
            "lane_count": 2,
            "duration": 180000,
            "bpm": None,
            "notes": [
                {
                    "lane_count": 2,
                    "notes": [{"time": 1000, "lane": 0, "energy": 0.5, "duration": None, "note_type": "drum"}],
                },
                {"lane_count": 2, "notes": []},
            ],
            "created_at": now,
        }
    )
    beatmap_id = str(beatmap_result.inserted_id)
    await fake_db.map_jobs.update_one({"_id": job_result.inserted_id}, {"$set": {"beatmap_id": beatmap_id}})

    job_response = await client.get(f"/api/v1/maps/jobs/{job_result.inserted_id}")
    assert job_response.status_code == 200
    assert job_response.json() == {
        "job_id": str(job_result.inserted_id),
        "status": "completed",
        "beatmap_id": beatmap_id,
        "error": None,
    }

    beatmap_response = await client.get(f"/api/v1/maps/{beatmap_id}")
    assert beatmap_response.status_code == 200
    beatmap_body = beatmap_response.json()
    assert beatmap_body["id"] == beatmap_id
    assert beatmap_body["lane_count"] == 2
    assert beatmap_body["notes"] == [
        {
            "lane_count": 2,
            "notes": [{"time": 1000, "lane": 0, "energy": 0.5, "duration": None, "note_type": "drum"}],
        },
        {"lane_count": 2, "notes": []},
    ]
    # no MINIO_PUBLIC_BASE_URL configured in test_config, so no CDN URL to hand back
    assert beatmap_body["audio_url"] is None


async def test_get_beatmap_returns_audio_url_when_cdn_configured(client, fake_db, test_maps_storage_config):
    test_maps_storage_config.public_base_url = "https://cdn.example.com"
    now = datetime.now(UTC)
    beatmap_result = await fake_db.beatmaps.insert_one(
        {
            "job_id": "job-1",
            "object_key": "job-1/original/song.mp3",
            "title": "song.mp3",
            "lane_count": 2,
            "duration": 180000,
            "bpm": None,
            "notes": [{"lane_count": 2, "notes": []}, {"lane_count": 2, "notes": []}],
            "created_at": now,
        }
    )

    response = await client.get(f"/api/v1/maps/{beatmap_result.inserted_id}")

    assert response.status_code == 200
    assert response.json()["audio_url"] == "https://cdn.example.com/job-1/original/song.mp3"


async def test_list_beatmaps_returns_newest_first(client, fake_db):
    older = await fake_db.beatmaps.insert_one(
        {
            "job_id": "job-1",
            "title": "older.mp3",
            "lane_count": 2,
            "duration": 100000,
            "bpm": None,
            "notes": [{"lane_count": 2, "notes": []}, {"lane_count": 2, "notes": []}],
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
    )
    newer = await fake_db.beatmaps.insert_one(
        {
            "job_id": "job-2",
            "title": "newer.mp3",
            "lane_count": 2,
            "duration": 200000,
            "bpm": None,
            "notes": [{"lane_count": 2, "notes": []}, {"lane_count": 2, "notes": []}],
            "created_at": datetime(2026, 2, 1, tzinfo=UTC),
        }
    )

    response = await client.get("/api/v1/maps")

    assert response.status_code == 200
    body = response.json()
    assert [entry["id"] for entry in body] == [str(newer.inserted_id), str(older.inserted_id)]


async def test_list_beatmaps_empty(client):
    response = await client.get("/api/v1/maps")

    assert response.status_code == 200
    assert response.json() == []
