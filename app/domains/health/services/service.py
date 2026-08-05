import redis.asyncio as redis
from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.db import get_db
from app.core.redis_client import get_redis


class HealthService:
    """Checks whether this instance's Mongo and Redis connections are actually reachable."""

    def __init__(self, db: AsyncIOMotorDatabase, redis_client: redis.Redis) -> None:
        self._db = db
        self._redis_client = redis_client

    async def check_readiness(self) -> dict:
        mongo_ok = await self._ping_mongo()
        redis_ok = await self._ping_redis()
        overall = "ok" if mongo_ok and redis_ok else "degraded"
        return {"status": overall, "mongo": mongo_ok, "redis": redis_ok}

    async def _ping_mongo(self) -> bool:
        try:
            await self._db.command("ping")
            return True
        except Exception:
            return False

    async def _ping_redis(self) -> bool:
        try:
            await self._redis_client.ping()
            return True
        except Exception:
            return False


def get_health_service(
    db: AsyncIOMotorDatabase = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
) -> HealthService:
    return HealthService(db, redis_client)
