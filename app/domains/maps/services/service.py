from io import BytesIO

from bullmq import FlowProducer
from fastapi import Depends, UploadFile

from app.core.config import Config, get_config
from app.domains.maps.dtos.api import Beatmap, MapJobCreateResponse, MapJobOut
from app.domains.maps.dtos.internal import NewMapJob
from app.domains.maps.exceptions.exceptions import (
    BeatmapNotFoundError,
    InvalidLaneCountError,
    MapJobNotFoundError,
    UnsupportedAudioTypeError,
)
from app.domains.maps.jobs.queues.queue import build_generate_beatmap_flow, get_flow_producer
from app.domains.maps.repositories.repository import MapRepository, get_map_repository
from app.domains.maps.services.storage import MapsStorage, get_maps_storage


class MapService:
    """Orchestrates map-job creation and lookup across Mongo, MinIO, and the job flow producer."""

    def __init__(
        self,
        repository: MapRepository,
        maps_storage: MapsStorage,
        flow_producer: FlowProducer,
        config: Config,
    ) -> None:
        self._repository = repository
        self._maps_storage = maps_storage
        self._flow_producer = flow_producer
        self._config = config

    async def create_map_job(self, *, file: UploadFile, lane_count: int | None) -> MapJobCreateResponse:
        content_type = file.content_type or ""
        if content_type not in self._config.allowed_audio_content_types:
            raise UnsupportedAudioTypeError(
                f"unsupported content type '{content_type}', expected one of "
                f"{self._config.allowed_audio_content_types}"
            )

        resolved_lane_count = lane_count if lane_count is not None else self._config.default_lane_count
        if not (1 <= resolved_lane_count <= self._config.max_lane_count):
            raise InvalidLaneCountError(
                f"lane_count must be between 1 and {self._config.max_lane_count}, got {resolved_lane_count}"
            )

        audio_bytes = await file.read()

        original_filename = file.filename or "audio"
        job_id = await self._repository.insert_map_job(
            NewMapJob(original_filename=original_filename, object_key=None, lane_count=resolved_lane_count)
        )
        object_key = self._maps_storage.original_key(job_id, original_filename)
        await self._repository.set_job_object_key(job_id, object_key)

        # object_key is scoped by job_id and never overwritten, so it's safe for a CDN to
        # cache the response forever - see MapsStorageConfig.cache_control.
        await self._maps_storage.upload(object_key, BytesIO(audio_bytes))
        flow = build_generate_beatmap_flow(
            job_id=job_id,
            object_key=object_key,
            original_filename=original_filename,
            lane_count=resolved_lane_count,
        )
        await self._flow_producer.add(flow)

        return MapJobCreateResponse(job_id=job_id, status="queued")

    async def get_job_status(self, job_id: str) -> MapJobOut:
        job = await self._repository.get_map_job(job_id)
        if job is None:
            raise MapJobNotFoundError(f"no map job with id '{job_id}'")
        return MapJobOut(
            job_id=str(job["_id"]),
            status=job["status"],
            beatmap_id=job.get("beatmap_id"),
            error=job.get("error"),
        )

    async def get_beatmap(self, beatmap_id: str) -> Beatmap:
        beatmap = await self._repository.get_beatmap(beatmap_id)
        if beatmap is None:
            raise BeatmapNotFoundError(f"no beatmap with id '{beatmap_id}'")
        return self._to_beatmap(beatmap)

    async def list_beatmaps(self) -> list[Beatmap]:
        beatmaps = await self._repository.list_beatmaps()
        return [self._to_beatmap(beatmap) for beatmap in beatmaps]

    def _to_beatmap(self, beatmap: dict) -> Beatmap:
        return Beatmap(
            id=str(beatmap["_id"]),
            title=beatmap["title"],
            lane_count=beatmap["lane_count"],
            duration_ms=beatmap["duration_ms"],
            bpm=beatmap.get("bpm"),
            notes=beatmap["notes"],
            created_at=beatmap["created_at"],
            audio_url=self._maps_storage.audio_url(beatmap.get("object_key")),
        )


def get_map_service(
    repository: MapRepository = Depends(get_map_repository),
    maps_storage: MapsStorage = Depends(get_maps_storage),
    flow_producer: FlowProducer = Depends(get_flow_producer),
    config: Config = Depends(get_config),
) -> MapService:
    """FastAPI dependency: a MapService assembled from this request's injected dependencies."""
    return MapService(repository, maps_storage, flow_producer, config)
