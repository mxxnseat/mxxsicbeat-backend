import redis.asyncio as redis
from fastapi import Request


def get_redis(request: Request) -> redis.Redis:
    """FastAPI dependency: the Redis client bound to this app instance's lifespan."""
    return request.app.state.redis
