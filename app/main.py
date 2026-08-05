from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.common.exceptions import register_exception_handlers
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.core.storage import Storage
from app.domains.health.routes.router import router as health_router
from app.domains.maps.configs.storage import get_maps_storage_config
from app.domains.maps.jobs.queues.queue import create_flow_producer
from app.domains.maps.routes.router import router as maps_router
from app.domains.maps.services.storage import MapsStorage

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    configure_logging(config.log_level)

    mongo_client = create_mongo_client(config)
    db = get_database(mongo_client, config)

    storage = Storage(config)
    maps_storage = MapsStorage(storage, get_maps_storage_config())
    await maps_storage.ensure_bucket()

    redis_client = redis.Redis.from_url(config.redis_url)
    flow_producer = create_flow_producer(config)

    app.state.mongo_client = mongo_client
    app.state.db = db
    app.state.storage = storage
    app.state.redis = redis_client
    app.state.flow_producer = flow_producer

    logger.info("app.started", env=config.env)
    try:
        yield
    finally:
        await flow_producer.close()
        await redis_client.aclose()
        mongo_client.close()
        logger.info("app.stopped")


def create_app() -> FastAPI:
    config = get_config()
    app = FastAPI(title="mxxsicbeat-backend", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(maps_router)

    return app


app = create_app()
