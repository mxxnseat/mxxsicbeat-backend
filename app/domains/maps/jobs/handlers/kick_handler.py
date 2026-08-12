import asyncio
import tempfile
from pathlib import Path

from bullmq import Job

from app.common.job_retry import is_final_attempt
from app.common.worker_runtime import WorkerRuntime
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.core.storage import Storage
from app.domains.maps.configs.storage import get_maps_storage_config
from app.domains.maps.dsp.kick_detector import detect_kick_onsets, has_significant_drum_energy
from app.domains.maps.jobs.queues.queue import (
    KICK_ONSET_QUEUE_NAME,
    DetectKickOnsetsJobPayload,
    KickDetectionResult,
)
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService
from app.domains.maps.services.notes_service import build_drum_notes
from app.domains.maps.services.storage import MapsStorage

logger = get_logger(__name__)


class KickHandler:
    def __init__(self, job_service: MapJobService, maps_storage: MapsStorage) -> None:
        self._job_service = job_service
        self._maps_storage = maps_storage

    async def __call__(self, job: Job, token: str) -> dict:
        try:
            return await self._process(job)
        except Exception as exc:
            job_id = job.data.get("job_id")
            if is_final_attempt(job):
                logger.error("kick_handler.failed", job_id=job_id, error=str(exc))
                await self._job_service.mark_failed(job_id, str(exc))
            else:
                logger.warning(
                    "kick_handler.retrying", job_id=job_id, error=str(exc), attempts_made=job.attemptsMade
                )
            raise

    async def _process(self, job: Job) -> dict:
        payload = DetectKickOnsetsJobPayload.model_validate(job.data)

        logger.info("kick_handler.processing", job_id=payload.job_id)

        with tempfile.TemporaryDirectory(prefix="mxxsicbeat-kick-") as tmp:
            drum_stem_path = Path(tmp) / "drums.wav"
            original_path = Path(tmp) / payload.original_filename
            await self._maps_storage.download(self._maps_storage.drum_key(payload.job_id), drum_stem_path)
            await self._maps_storage.download(
                self._maps_storage.original_key(payload.job_id, payload.original_filename), original_path
            )

            if not await asyncio.to_thread(has_significant_drum_energy, drum_stem_path, original_path):
                logger.info("kick_handler.no_significant_drums", job_id=payload.job_id)
                return KickDetectionResult(notes=[]).model_dump()

            onset_times, onseter = await asyncio.to_thread(detect_kick_onsets, drum_stem_path)

        notes = build_drum_notes(onset_times, onseter, payload.lane_count)

        logger.info("kick_handler.completed", job_id=payload.job_id)
        return KickDetectionResult(notes=notes).model_dump()


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
