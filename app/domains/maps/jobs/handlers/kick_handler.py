import asyncio
import tempfile
from pathlib import Path

from bullmq import Job

from app.common.worker_runtime import WorkerRuntime
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.core.storage import Storage
from app.domains.maps.configs.storage import get_maps_storage_config
from app.domains.maps.dsp.pipeline import generate_beatmap
from app.domains.maps.jobs.queues.queue import KICK_ONSET_QUEUE_NAME, DetectKickOnsetsJobPayload
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService
from app.domains.maps.services.storage import MapsStorage

logger = get_logger(__name__)


class KickHandler:
    """BullMQ job processor: downloads audio, runs kick-onset detection, and returns the
    resulting notes as the job's result - the parent `beatmap_handler` reads this back via
    `job.getChildrenValues()` once this and `midi_handler` have both completed.
    """

    def __init__(self, job_service: MapJobService, maps_storage: MapsStorage) -> None:
        self._job_service = job_service
        self._maps_storage = maps_storage

    async def __call__(self, job: Job, token: str) -> dict:
        try:
            return await self._process(job)
        except Exception as exc:
            job_id = job.data.get("job_id")
            logger.error("kick_handler.failed", job_id=job_id, error=str(exc))
            await self._job_service.mark_failed(job_id, str(exc))
            raise

    async def _process(self, job: Job) -> dict:
        payload = DetectKickOnsetsJobPayload.model_validate(job.data)

        await self._job_service.mark_processing(payload.job_id)
        logger.info("kick_handler.processing", job_id=payload.job_id, object_key=payload.object_key)

        with tempfile.TemporaryDirectory(prefix="mxxsicbeat-kick-") as tmp:
            work_dir = Path(tmp)
            audio_path = work_dir / payload.original_filename

            await self._maps_storage.download(payload.object_key, audio_path)

            result = await asyncio.to_thread(generate_beatmap, audio_path, work_dir, payload.lane_count)

            with result.drum_stem_path.open("rb") as drum_stem:
                await self._maps_storage.upload(self._maps_storage.drum_key(payload.job_id), drum_stem)
            with result.melody_stem_path.open("rb") as melody_stem:
                await self._maps_storage.upload(self._maps_storage.melody_key(payload.job_id), melody_stem)

        logger.info("kick_handler.completed", job_id=payload.job_id)
        return {"lane_count": result.lane_count, "duration_ms": result.duration_ms, "notes": result.notes}


async def main() -> None:
    config = get_config()
    configure_logging(config.log_level)

    mongo_client = create_mongo_client(config)
    db = get_database(mongo_client, config)
    job_service = MapJobService(MapRepository(db))
    maps_storage = MapsStorage(Storage(config), get_maps_storage_config())

    processor = KickHandler(job_service, maps_storage)
    runtime = WorkerRuntime(config, KICK_ONSET_QUEUE_NAME, processor, on_stop=[mongo_client.close])
    await runtime.start()
    try:
        await runtime.run_until_stopped()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
