from fastapi import Request
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Config


def create_mongo_client(config: Config) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(config.mongodb_uri)


def get_database(client: AsyncIOMotorClient, config: Config) -> AsyncIOMotorDatabase:
    return client.get_database(config.mongodb_db_name)


def get_db(request: Request) -> AsyncIOMotorDatabase:
    return request.app.state.db
