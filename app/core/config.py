from functools import lru_cache

from pydantic_settings import SettingsConfigDict

from app.configs.app import AppConfig
from app.configs.minio import MinioConfig
from app.configs.mongo import MongoConfig
from app.configs.redis import RedisConfig
from app.configs.uploads import UploadConfig


class Config(AppConfig, MongoConfig, RedisConfig, MinioConfig, UploadConfig):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_config() -> Config:
    return Config()
