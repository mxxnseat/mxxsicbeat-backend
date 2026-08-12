import asyncio
import tempfile
from pathlib import Path

import librosa
from bullmq import Job

from app.common.worker_runtime import WorkerRuntime
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.core.storage import Storage
from app.domains.maps.configs.storage import get_maps_storage_config
from app.domains.maps.jobs.queues.queue import (
    MELODY_EXTRACTION_QUEUE_NAME,
    ExtractMelodyJobPayload,
    MelodyExtractionResult,
)
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService
from app.domains.maps.services.notes_service import build_melody_notes
from app.domains.maps.services.storage import MapsStorage

logger = get_logger(__name__)


class MelodyHandler:
    """BullMQ job processor: downloads the melody stem stem_handler already separated out
    (drums and vocals removed) and runs onset + spectral-centroid-lane detection directly
    against it, returning the resulting notes - the parent `beatmap_handler` reads this back via
    `job.getChildrenValues()` once this and `kick_handler` have both completed.
    """

    def __init__(self, job_service: MapJobService, maps_storage: MapsStorage) -> None:
        self._job_service = job_service
        self._maps_storage = maps_storage

    async def __call__(self, job: Job, token: str) -> dict:
        try:
            return await self._process(job)
        except Exception as exc:
            job_id = job.data.get("job_id")
            logger.error("melody_handler.failed", job_id=job_id, error=str(exc))
            await self._job_service.mark_failed(job_id, str(exc))
            raise

    async def _process(self, job: Job) -> dict:
        payload = ExtractMelodyJobPayload.model_validate(job.data)

        logger.info("melody_handler.processing", job_id=payload.job_id)

        with tempfile.TemporaryDirectory(prefix="mxxsicbeat-melody-") as tmp:
            melody_stem_path = Path(tmp) / "melody.wav"
            await self._maps_storage.download(
                self._maps_storage.melody_key(payload.job_id), melody_stem_path
            )

            y, sr = await asyncio.to_thread(librosa.load, melody_stem_path, sr=None, mono=True)

        notes = build_melody_notes(y, int(sr), payload.bpm, payload.lane_count)

        logger.info("melody_handler.completed", job_id=payload.job_id)
        return MelodyExtractionResult(notes=notes).model_dump()


async def main() -> None:
    config = get_config()
    configure_logging(config.log_level)

    mongo_client = create_mongo_client(config)
    db = get_database(mongo_client, config)
    job_service = MapJobService(MapRepository(db))
    maps_storage = MapsStorage(Storage(config), get_maps_storage_config())

    processor = MelodyHandler(job_service, maps_storage)
    runtime = WorkerRuntime(config, MELODY_EXTRACTION_QUEUE_NAME, processor, on_stop=[mongo_client.close])
    await runtime.start()
    try:
        await runtime.run_until_stopped()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
