from bullmq import FlowProducer
from fastapi import Request
from pydantic import BaseModel

from app.core.config import Config
from app.domains.maps.configs.job import (
    get_beatmap_orchestration_job_config,
    get_kick_onset_job_config,
    get_melody_extraction_job_config,
    get_stem_separation_job_config,
)
from app.domains.maps.dtos.notes import Note

STEM_SEPARATION_QUEUE_NAME = "stem-separation"
SEPARATE_STEMS_JOB_NAME = "separate-stems"

BEATMAP_ORCHESTRATION_QUEUE_NAME = "beatmap-orchestration"
ORCHESTRATE_BEATMAP_JOB_NAME = "orchestrate-beatmap"

KICK_ONSET_QUEUE_NAME = "kick-onset-detection"
DETECT_KICK_ONSETS_JOB_NAME = "detect-kick-onsets"

MELODY_EXTRACTION_QUEUE_NAME = "melody-extraction"
EXTRACT_MELODY_JOB_NAME = "extract-melody"


class SeparateStemsJobPayload(BaseModel):
    job_id: str
    object_key: str
    original_filename: str
    lane_count: int


class OrchestrateBeatmapJobPayload(BaseModel):
    job_id: str
    object_key: str
    original_filename: str
    lane_count: int
    duration: int
    bpm: int


class DetectKickOnsetsJobPayload(BaseModel):
    job_id: str
    lane_count: int
    object_key: str
    original_filename: str


class ExtractMelodyJobPayload(BaseModel):
    job_id: str
    lane_count: int
    bpm: int


class KickDetectionResult(BaseModel):
    notes: list[Note]


class MelodyExtractionResult(BaseModel):
    notes: list[Note]


def build_separate_stems_job(
    *, job_id: str, object_key: str, original_filename: str, lane_count: int
) -> dict:
    payload = SeparateStemsJobPayload(
        job_id=job_id, object_key=object_key, original_filename=original_filename, lane_count=lane_count
    )
    return {
        "name": SEPARATE_STEMS_JOB_NAME,
        "queueName": STEM_SEPARATION_QUEUE_NAME,
        "data": payload.model_dump(),
        "opts": get_stem_separation_job_config().to_bullmq_opts(),
    }


def build_generate_beatmap_flow(
    *, job_id: str, object_key: str, original_filename: str, lane_count: int, duration: int, bpm: int
) -> dict:
    kick_payload = DetectKickOnsetsJobPayload(
        job_id=job_id, lane_count=lane_count, object_key=object_key, original_filename=original_filename
    )
    melody_payload = ExtractMelodyJobPayload(job_id=job_id, lane_count=lane_count, bpm=bpm)
    root_payload = OrchestrateBeatmapJobPayload(
        job_id=job_id,
        object_key=object_key,
        original_filename=original_filename,
        lane_count=lane_count,
        duration=duration,
        bpm=bpm,
    )

    return {
        "name": ORCHESTRATE_BEATMAP_JOB_NAME,
        "queueName": BEATMAP_ORCHESTRATION_QUEUE_NAME,
        "data": root_payload.model_dump(),
        "opts": get_beatmap_orchestration_job_config().to_bullmq_opts(),
        "children": [
            {
                "name": DETECT_KICK_ONSETS_JOB_NAME,
                "queueName": KICK_ONSET_QUEUE_NAME,
                "data": kick_payload.model_dump(),
                "opts": {"failParentOnFailure": True, **get_kick_onset_job_config().to_bullmq_opts()},
            },
            {
                "name": EXTRACT_MELODY_JOB_NAME,
                "queueName": MELODY_EXTRACTION_QUEUE_NAME,
                "data": melody_payload.model_dump(),
                "opts": {"failParentOnFailure": True, **get_melody_extraction_job_config().to_bullmq_opts()},
            },
        ],
    }


def create_flow_producer(config: Config) -> FlowProducer:
    return FlowProducer({"connection": config.redis_url})


def get_flow_producer(request: Request) -> FlowProducer:
    return request.app.state.flow_producer
