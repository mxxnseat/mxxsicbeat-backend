from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class MapsStorageConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="MAPS_STORAGE_", extra="ignore"
    )

    songs_bucket: str = "songs"

    original_dir: str = "original"
    separated_dir: str = "separated"

    drum_filename: str = "drum.wav"
    melody_filename: str = "melody.wav"

    cache_control: str = "public, max-age=31536000, immutable"

    public_base_url: str | None = None

    def original_key(self, job_id: str, filename: str) -> str:
        return f"{job_id}/{self.original_dir}/{filename}"

    def drum_key(self, job_id: str) -> str:
        return f"{job_id}/{self.separated_dir}/{self.drum_filename}"

    def melody_key(self, job_id: str) -> str:
        return f"{job_id}/{self.separated_dir}/{self.melody_filename}"


@lru_cache
def get_maps_storage_config() -> MapsStorageConfig:
    return MapsStorageConfig()
