from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class MapsJobConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    attempts: int = 3
    backoff_type: Literal["fixed", "exponential"] = "exponential"
    backoff_delay_ms: int = 5000

    def to_bullmq_opts(self) -> dict:
        backoff = {"type": self.backoff_type, "delay": self.backoff_delay_ms}
        return {"attempts": self.attempts, "backoff": backoff}


class StemSeparationJobConfig(MapsJobConfig):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="MAPS_JOB_STEM_SEPARATION_", extra="ignore"
    )


class KickOnsetJobConfig(MapsJobConfig):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="MAPS_JOB_KICK_ONSET_", extra="ignore"
    )


class MelodyExtractionJobConfig(MapsJobConfig):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="MAPS_JOB_MELODY_EXTRACTION_", extra="ignore"
    )


class BeatmapOrchestrationJobConfig(MapsJobConfig):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MAPS_JOB_BEATMAP_ORCHESTRATION_",
        extra="ignore",
    )


@lru_cache
def get_stem_separation_job_config() -> StemSeparationJobConfig:
    return StemSeparationJobConfig()


@lru_cache
def get_kick_onset_job_config() -> KickOnsetJobConfig:
    return KickOnsetJobConfig()


@lru_cache
def get_melody_extraction_job_config() -> MelodyExtractionJobConfig:
    return MelodyExtractionJobConfig()


@lru_cache
def get_beatmap_orchestration_job_config() -> BeatmapOrchestrationJobConfig:
    return BeatmapOrchestrationJobConfig()
