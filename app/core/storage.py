from pathlib import Path
from typing import IO

import aioboto3
from botocore.exceptions import ClientError
from fastapi import Request

from app.core.config import Config


class Storage:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._session = aioboto3.Session()

    def _client(self):
        return self._session.client(
            "s3",
            endpoint_url=self._config.minio_endpoint_url,
            aws_access_key_id=self._config.minio_access_key,
            aws_secret_access_key=self._config.minio_secret_key,
            region_name=self._config.minio_region,
        )

    async def ensure_bucket(self, bucket: str) -> None:
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=bucket)
            except ClientError:
                await s3.create_bucket(Bucket=bucket)

    async def upload_fileobj(
        self,
        bucket: str,
        key: str,
        fileobj: IO[bytes],
        content_type: str | None = None,
        cache_control: str | None = None,
    ) -> None:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if cache_control:
            extra_args["CacheControl"] = cache_control
        async with self._client() as s3:
            await s3.upload_fileobj(fileobj, bucket, key, ExtraArgs=extra_args)

    async def download_to_path(self, bucket: str, key: str, destination: Path) -> None:
        async with self._client() as s3:
            await s3.download_file(bucket, key, str(destination))


def get_storage(request: Request) -> Storage:
    return request.app.state.storage
