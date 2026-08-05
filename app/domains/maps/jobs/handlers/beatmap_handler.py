import asyncio

from bullmq import Job

from app.common.worker_runtime import WorkerRuntime
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.domains.maps.dtos.internal import NewBeatmap
from app.domains.maps.jobs.queues.queue import (
    BEATMAP_ORCHESTRATION_QUEUE_NAME,
    KICK_ONSET_QUEUE_NAME,
    OrchestrateBeatmapJobPayload,
)
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService

logger = get_logger(__name__)


def _child_result(children: dict, queue_name: str) -> dict:
    """`Job.getChildrenValues()` returns a dict keyed by `<prefix>:<queueName>:<jobId>` mapping
    to each child's JSON-decoded return value - match by queue name rather than assuming key
    order or exact format."""

    for key, value in children.items():
        if queue_name in key:
            return value
    raise KeyError(f"no child result found for queue '{queue_name}'")


class BeatmapHandler:
    """Orchestrator/root processor for the beatmap-generation flow. BullMQ only invokes this
    once both children - `kick_handler` and `midi_handler`, which run in parallel - have
    completed, so this never touches audio itself: it just merges their results and saves the
    finished beatmap.
    """

    def __init__(self, job_service: MapJobService) -> None:
        self._job_service = job_service

    async def __call__(self, job: Job, token: str) -> dict:
        job_id = job.data.get("job_id")
        try:
            return await self._process(job)
        except Exception as exc:
            logger.error("beatmap_handler.failed", job_id=job_id, error=str(exc))
            await self._job_service.mark_failed(job_id, str(exc))
            raise

    async def _process(self, job: Job) -> dict:
        payload = OrchestrateBeatmapJobPayload.model_validate(job.data)

        children = await job.getChildrenValues()
        kick_result = _child_result(children, KICK_ONSET_QUEUE_NAME)
        # midi_result = _child_result(children, MIDI_EXTRACTION_QUEUE_NAME) - always {"notes": []}
        # for now; reserved as the merge point once midi_handler does real extraction.

        new_beatmap = NewBeatmap(
            job_id=payload.job_id,
            object_key=payload.object_key,
            title=payload.original_filename,
            lane_count=kick_result["lane_count"],
            duration_ms=kick_result["duration_ms"],
            bpm=None,
            notes=kick_result["notes"],
        )
        beatmap_id = await self._job_service.finalize_beatmap(new_beatmap)
        logger.info("beatmap_handler.completed", job_id=payload.job_id, beatmap_id=beatmap_id)
        return {"beatmap_id": beatmap_id}


async def main() -> None:
    config = get_config()
    configure_logging(config.log_level)

    mongo_client = create_mongo_client(config)
    db = get_database(mongo_client, config)
    job_service = MapJobService(MapRepository(db))

    processor = BeatmapHandler(job_service)
    runtime = WorkerRuntime(
        config, BEATMAP_ORCHESTRATION_QUEUE_NAME, processor, on_stop=[mongo_client.close]
    )
    await runtime.start()
    try:
        await runtime.run_until_stopped()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
