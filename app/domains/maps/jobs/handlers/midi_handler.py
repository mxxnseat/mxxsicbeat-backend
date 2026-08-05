import asyncio

from bullmq import Job

from app.common.worker_runtime import WorkerRuntime
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.domains.maps.jobs.queues.queue import MIDI_EXTRACTION_QUEUE_NAME, ExtractMidiJobPayload
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService

logger = get_logger(__name__)


class MidiHandler:
    """BullMQ job processor for MIDI/pitch extraction.

    Placeholder: no extraction is implemented yet (see CLAUDE.md - "bpm and per-note pitch are
    reserved for a future pass wiring in basic-pitch"), so this always succeeds with an empty
    note list rather than downloading audio it wouldn't do anything with. It must never fail the
    beatmap-generation flow while unimplemented - failParentOnFailure means a failure here would
    also fail `beatmap_handler`.

    TODO: wire in basic-pitch here once that DSP work lands.
    """

    def __init__(self, job_service: MapJobService) -> None:
        self._job_service = job_service

    async def __call__(self, job: Job, token: str) -> dict:
        try:
            payload = ExtractMidiJobPayload.model_validate(job.data)
        except Exception as exc:
            job_id = job.data.get("job_id")
            logger.error("midi_handler.failed", job_id=job_id, error=str(exc))
            await self._job_service.mark_failed(job_id, str(exc))
            raise

        logger.info("midi_handler.skipped", job_id=payload.job_id)
        return {"notes": []}


async def main() -> None:
    config = get_config()
    configure_logging(config.log_level)

    mongo_client = create_mongo_client(config)
    db = get_database(mongo_client, config)
    job_service = MapJobService(MapRepository(db))

    processor = MidiHandler(job_service)
    runtime = WorkerRuntime(config, MIDI_EXTRACTION_QUEUE_NAME, processor, on_stop=[mongo_client.close])
    await runtime.start()
    try:
        await runtime.run_until_stopped()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
