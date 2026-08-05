from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "mxxsicbeat"
