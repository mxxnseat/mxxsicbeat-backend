import asyncio

from bullmq import Job

from app.common.job_retry import is_final_attempt
from app.common.worker_runtime import WorkerRuntime
from app.core.config import get_config
from app.core.db import create_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.domains.maps.dtos.internal import NewBeatmap
from app.domains.maps.dtos.notes import NoteGroup
from app.domains.maps.jobs.queues.queue import (
    BEATMAP_ORCHESTRATION_QUEUE_NAME,
    KICK_ONSET_QUEUE_NAME,
    MELODY_EXTRACTION_QUEUE_NAME,
    KickDetectionResult,
    MelodyExtractionResult,
    OrchestrateBeatmapJobPayload,
)
from app.domains.maps.repositories.repository import MapRepository
from app.domains.maps.services.job_service import MapJobService

logger = get_logger(__name__)


def _child_result(children: dict, queue_name: str) -> dict:
    for key, value in children.items():
        if queue_name in key:
            return value
    raise KeyError(f"no child result found for queue '{queue_name}'")


class BeatmapHandler:
    def __init__(self, job_service: MapJobService) -> None:
        self._job_service = job_service

    async def __call__(self, job: Job, token: str) -> dict:
        job_id = job.data.get("job_id")
        try:
            return await self._process(job)
        except Exception as exc:
            if is_final_attempt(job):
                logger.error("beatmap_handler.failed", job_id=job_id, error=str(exc))
                await self._job_service.mark_failed(job_id, str(exc))
            else:
                logger.warning(
                    "beatmap_handler.retrying", job_id=job_id, error=str(exc), attempts_made=job.attemptsMade
                )
            raise

    async def _process(self, job: Job) -> dict:
        payload = OrchestrateBeatmapJobPayload.model_validate(job.data)

        children = await job.getChildrenValues()
        kick_result = KickDetectionResult.model_validate(_child_result(children, KICK_ONSET_QUEUE_NAME))
        melody_result = MelodyExtractionResult.model_validate(_child_result(children, MELODY_EXTRACTION_QUEUE_NAME))

        new_beatmap = NewBeatmap(
            job_id=payload.job_id,
            object_key=payload.object_key,
            title=payload.original_filename,
            duration=payload.duration,
            bpm=payload.bpm,
            notes=self.generate_notes(payload.lane_count, kick_result, melody_result),
        )
        beatmap_id = await self._job_service.finalize_beatmap(new_beatmap)
        logger.info("beatmap_handler.completed", job_id=payload.job_id, beatmap_id=beatmap_id)
        return {"beatmap_id": beatmap_id}

    def generate_notes(
        self, lane_count: int, kick_result: KickDetectionResult, melody_result: MelodyExtractionResult
    ) -> tuple[NoteGroup, NoteGroup]:
        return (
            NoteGroup(lane_count=lane_count, notes=list(kick_result.notes)),
            NoteGroup(lane_count=lane_count, notes=list(melody_result.notes)),
        )

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
