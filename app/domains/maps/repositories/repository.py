from datetime import UTC, datetime

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.db import get_db
from app.domains.maps.dtos.internal import NewBeatmap, NewMapJob


def to_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


class MapRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    async def insert_map_job(self, job: NewMapJob) -> str:
        now = datetime.now(UTC)
        result = await self._db.map_jobs.insert_one(
            {
                "status": "queued",
                "original_filename": job.original_filename,
                "object_key": job.object_key,
                "lane_count": job.lane_count,
                "beatmap_id": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        return str(result.inserted_id)

    async def get_map_job(self, job_id: str) -> dict | None:
        object_id = to_object_id(job_id)
        if object_id is None:
            return None
        return await self._db.map_jobs.find_one({"_id": object_id})

    async def set_job_object_key(self, job_id: str, object_key: str) -> None:
        await self._db.map_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"object_key": object_key, "updated_at": datetime.now(UTC)}},
        )

    async def mark_job_processing(self, job_id: str) -> None:
        await self._db.map_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "processing", "updated_at": datetime.now(UTC)}},
        )

    async def mark_job_completed(self, job_id: str, beatmap_id: str) -> None:
        await self._db.map_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {
                "$set": {
                    "status": "completed",
                    "beatmap_id": beatmap_id,
                    "updated_at": datetime.now(UTC),
                }
            },
        )

    async def mark_job_failed(self, job_id: str, error: str) -> None:
        await self._db.map_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "failed", "error": error, "updated_at": datetime.now(UTC)}},
        )

    async def insert_beatmap(self, beatmap: NewBeatmap) -> str:
        result = await self._db.beatmaps.insert_one(
            {
                "job_id": beatmap.job_id,
                "object_key": beatmap.object_key,
                "title": beatmap.title,
                "duration": beatmap.duration,
                "bpm": beatmap.bpm,
                "notes": [group.model_dump() for group in beatmap.notes],
                "created_at": datetime.now(UTC),
            }
        )
        return str(result.inserted_id)

    async def get_beatmap(self, beatmap_id: str) -> dict | None:
        object_id = to_object_id(beatmap_id)
        if object_id is None:
            return None
        return await self._db.beatmaps.find_one({"_id": object_id})

    async def list_beatmaps(self) -> list[dict]:
        cursor = self._db.beatmaps.find().sort("created_at", -1)
        return await cursor.to_list(length=None)


def get_map_repository(db: AsyncIOMotorDatabase = Depends(get_db)) -> MapRepository:
    return MapRepository(db)
