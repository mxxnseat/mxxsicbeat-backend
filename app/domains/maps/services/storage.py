import mimetypes
from pathlib import Path
from typing import IO

from fastapi import Depends

from app.core.storage import Storage, get_storage
from app.domains.maps.configs.storage import MapsStorageConfig, get_maps_storage_config


class MapsStorage:
    """Facade over the generic `Storage` client, bound to the maps domain's bucket and key
    layout. Callers only ever pass a key and a file object - bucket, content type (inferred
    from the key's extension) and cache control all come from `MapsStorageConfig`."""

    def __init__(self, storage: Storage, config: MapsStorageConfig) -> None:
        self._storage = storage
        self._config = config

    async def ensure_bucket(self) -> None:
        await self._storage.ensure_bucket(self._config.songs_bucket)

    async def upload(self, key: str, fileobj: IO[bytes]) -> None:
        content_type, _ = mimetypes.guess_type(key)
        await self._storage.upload_fileobj(
            self._config.songs_bucket,
            key,
            fileobj,
            content_type=content_type,
            cache_control=self._config.cache_control,
        )

    async def download(self, key: str, destination: Path) -> None:
        await self._storage.download_to_path(self._config.songs_bucket, key, destination)

    def original_key(self, job_id: str, filename: str) -> str:
        return self._config.original_key(job_id, filename)

    def drum_key(self, job_id: str) -> str:
        return self._config.drum_key(job_id)

    def melody_key(self, job_id: str) -> str:
        return self._config.melody_key(job_id)

    def audio_url(self, object_key: str | None) -> str | None:
        base_url = self._config.public_base_url
        if not base_url or not object_key:
            return None
        return f"{base_url.rstrip('/')}/{object_key}"


def get_maps_storage(
    storage: Storage = Depends(get_storage),
    config: MapsStorageConfig = Depends(get_maps_storage_config),
) -> MapsStorage:
    """FastAPI dependency: a MapsStorage facade assembled from this request's injected
    dependencies."""
    return MapsStorage(storage, config)
