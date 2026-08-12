from fastapi import Depends, File, UploadFile

from app.core.config import Config, get_config
from app.domains.maps.exceptions.exceptions import AudioTooLargeError


async def validate_file_size(
    file: UploadFile = File(..., description="Audio file to generate a beatmap from"),
    config: Config = Depends(get_config),
) -> UploadFile:
    max_bytes = config.max_upload_size_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise AudioTooLargeError(f"audio file exceeds the {config.max_upload_size_mb}MB limit")
    return file
