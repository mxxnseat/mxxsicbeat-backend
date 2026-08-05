from bullmq import FlowProducer
from fastapi import Request
from pydantic import BaseModel

from app.core.config import Config

BEATMAP_ORCHESTRATION_QUEUE_NAME = "beatmap-orchestration"
ORCHESTRATE_BEATMAP_JOB_NAME = "orchestrate-beatmap"

KICK_ONSET_QUEUE_NAME = "kick-onset-detection"
DETECT_KICK_ONSETS_JOB_NAME = "detect-kick-onsets"

MIDI_EXTRACTION_QUEUE_NAME = "midi-extraction"
EXTRACT_MIDI_JOB_NAME = "extract-midi"


class OrchestrateBeatmapJobPayload(BaseModel):
    """The root job's data - crosses a Redis/JSON boundary, so it's a validated pydantic model
    rather than a plain dict passed around by convention. Its processor never touches audio
    bytes, but it does carry object_key through so the finished beatmap doc can be linked back
    to its source file in MinIO (for building a CDN audio_url)."""

    job_id: str
    object_key: str
    original_filename: str
    lane_count: int


class DetectKickOnsetsJobPayload(BaseModel):
    job_id: str
    object_key: str
    original_filename: str
    lane_count: int


class ExtractMidiJobPayload(BaseModel):
    job_id: str
    object_key: str
    original_filename: str


def build_generate_beatmap_flow(
    *, job_id: str, object_key: str, original_filename: str, lane_count: int
) -> dict:
    """Builds the BullMQ Flow tree for FlowProducer.add(): a root orchestration job whose
    processor only becomes runnable once both children - kick onset detection and midi
    extraction, which run in parallel - have completed. `failParentOnFailure` on the children
    ensures a child that exhausts its retries also fails the root, instead of leaving it stuck
    waiting forever."""

    child_opts = {"failParentOnFailure": True}
    kick_payload = DetectKickOnsetsJobPayload(
        job_id=job_id,
        object_key=object_key,
        original_filename=original_filename,
        lane_count=lane_count,
    )
    midi_payload = ExtractMidiJobPayload(
        job_id=job_id, object_key=object_key, original_filename=original_filename
    )
    root_payload = OrchestrateBeatmapJobPayload(
        job_id=job_id, object_key=object_key, original_filename=original_filename, lane_count=lane_count
    )

    return {
        "name": ORCHESTRATE_BEATMAP_JOB_NAME,
        "queueName": BEATMAP_ORCHESTRATION_QUEUE_NAME,
        "data": root_payload.model_dump(),
        "children": [
            {
                "name": DETECT_KICK_ONSETS_JOB_NAME,
                "queueName": KICK_ONSET_QUEUE_NAME,
                "data": kick_payload.model_dump(),
                "opts": child_opts,
            },
            {
                "name": EXTRACT_MIDI_JOB_NAME,
                "queueName": MIDI_EXTRACTION_QUEUE_NAME,
                "data": midi_payload.model_dump(),
                "opts": child_opts,
            },
        ],
    }


def create_flow_producer(config: Config) -> FlowProducer:
    return FlowProducer({"connection": config.redis_url})


def get_flow_producer(request: Request) -> FlowProducer:
    """FastAPI dependency: the BullMQ flow producer bound to this app instance's lifespan."""
    return request.app.state.flow_producer
