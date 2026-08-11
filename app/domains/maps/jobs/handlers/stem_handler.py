import asyncio
import tempfile
from pathlib import Path

from bullmq import FlowProducer, Job

from app.common.worker_runtime import WorkerRuntime
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.core.storage import Storage
from app.domains.maps.configs.storage import get_maps_storage_config
from app.domains.maps.dsp.audio_metadata import detect_bpm, track_duration
from app.domains.maps.dsp.stem_separator import separate_stems
from app.domains.maps.jobs.queues.queue import (
    STEM_SEPARATION_QUEUE_NAME,
    SeparateStemsJobPayload,
    build_generate_beatmap_flow,
    create_flow_producer,
)
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService
from app.domains.maps.services.storage import MapsStorage

logger = get_logger(__name__)


class StemHandler:
    """BullMQ job processor: downloads the original upload and runs two-stage demucs separation
    (drums first, then vocals out of what's left), uploading the resulting drum/melody stems.
    On success it dynamically enqueues the second-phase flow - kick_handler and melody_handler
    in parallel, gated by beatmap_handler - since BullMQ jobs can only have a single parent and
    both of those need this job's output, they can't be modeled as its children directly.
    """

    def __init__(
        self, job_service: MapJobService, maps_storage: MapsStorage, flow_producer: FlowProducer
    ) -> None:
        self._job_service = job_service
        self._maps_storage = maps_storage
        self._flow_producer = flow_producer

    async def __call__(self, job: Job, token: str) -> dict:
        try:
            return await self._process(job)
        except Exception as exc:
            job_id = job.data.get("job_id")
            logger.error("stem_handler.failed", job_id=job_id, error=str(exc))
            await self._job_service.mark_failed(job_id, str(exc))
            raise

    async def _process(self, job: Job) -> dict:
        payload = SeparateStemsJobPayload.model_validate(job.data)

        await self._job_service.mark_processing(payload.job_id)
        logger.info("stem_handler.processing", job_id=payload.job_id, object_key=payload.object_key)

        with tempfile.TemporaryDirectory(prefix="mxxsicbeat-stems-") as tmp:
            work_dir = Path(tmp)
            audio_path = work_dir / payload.original_filename

            await self._maps_storage.download(payload.object_key, audio_path)

            # TODO: Don't read the same file twice in track_duration and detect_bpm
            duration = await asyncio.to_thread(track_duration, audio_path)
            bpm = await asyncio.to_thread(detect_bpm, audio_path)
            stems = await asyncio.to_thread(separate_stems, audio_path, work_dir)

            with stems.drum_stem_path.open("rb") as drum_stem:
                await self._maps_storage.upload(self._maps_storage.drum_key(payload.job_id), drum_stem)
            with stems.melody_stem_path.open("rb") as melody_stem:
                await self._maps_storage.upload(self._maps_storage.melody_key(payload.job_id), melody_stem)

        flow = build_generate_beatmap_flow(
            job_id=payload.job_id,
            object_key=payload.object_key,
            original_filename=payload.original_filename,
            lane_count=payload.lane_count,
            duration=duration,
            bpm=bpm,
        )
        await self._flow_producer.add(flow)

        logger.info("stem_handler.completed", job_id=payload.job_id)
        return {"job_id": payload.job_id}


async def main() -> None:
    config = get_config()
    configure_logging(config.log_level)

    mongo_client = create_mongo_client(config)
    db = get_database(mongo_client, config)
    job_service = MapJobService(MapRepository(db))
    maps_storage = MapsStorage(Storage(config), get_maps_storage_config())
    flow_producer = create_flow_producer(config)

    processor = StemHandler(job_service, maps_storage, flow_producer)
    runtime = WorkerRuntime(config, STEM_SEPARATION_QUEUE_NAME, processor, on_stop=[mongo_client.close])
    await runtime.start()
    try:
        await runtime.run_until_stopped()
    finally:
        await runtime.stop()
        await flow_producer.close()


if __name__ == "__main__":
    asyncio.run(main())
