from pydantic_settings import BaseSettings, SettingsConfigDict


class UploadConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    max_upload_size_mb: int = 30
    allowed_audio_content_types: list[str] = [
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
    ]
    default_lane_count: int = 2
    max_lane_count: int = 8
