"""Indexes backing job-status polling and beatmap lookup by job."""

from motor.motor_asyncio import AsyncIOMotorDatabase


async def up(db: AsyncIOMotorDatabase) -> None:
    await db.map_jobs.create_index("created_at")
    await db.beatmaps.create_index("job_id")
