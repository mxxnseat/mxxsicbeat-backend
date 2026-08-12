from bullmq import FlowProducer
from fastapi import Request
from pydantic import BaseModel

from app.core.config import Config
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
    """The stem-separation job's data - crosses a Redis/JSON boundary, so it's a validated
    pydantic model. Carries the same fields as the root beatmap-orchestration payload because
    stem_handler dynamically builds that second-phase flow itself once separation finishes."""

    job_id: str
    object_key: str
    original_filename: str
    lane_count: int


class OrchestrateBeatmapJobPayload(BaseModel):
    """The root job's data - crosses a Redis/JSON boundary, so it's a validated pydantic model
    rather than a plain dict passed around by convention. Its processor never touches audio
    bytes, but it does carry object_key through so the finished beatmap doc can be linked back
    to its source file in MinIO (for building a CDN audio_url). duration/bpm are track-level
    properties detected once by stem_handler from the original mix (the best available signal
    for both), not by kick_handler/melody_handler off their individual stems."""

    job_id: str
    object_key: str
    original_filename: str
    lane_count: int
    duration: int
    bpm: int


class DetectKickOnsetsJobPayload(BaseModel):
    """kick_handler downloads its drum stem from storage via the deterministic, job_id-keyed
    path in MapsStorageConfig, but also needs the original mix (object_key/original_filename)
    to compare against - see KickHandler for why."""

    job_id: str
    lane_count: int
    object_key: str
    original_filename: str


class ExtractMelodyJobPayload(BaseModel):
    """melody_handler downloads its melody stem from storage via the deterministic, job_id-keyed
    path in MapsStorageConfig - it never needs object_key/original_filename. bpm is threaded
    through from stem_handler so melody_handler can tag its MIDI output with the track's real
    tempo without re-detecting it from the (rhythm-section-free) melody stem."""

    job_id: str
    lane_count: int
    bpm: int


class KickDetectionResult(BaseModel):
    """kick_handler's job result - crosses the same Redis/JSON boundary as the payloads above,
    so it's a validated model rather than a hand-built dict. Read back by beatmap_handler via
    `job.getChildrenValues()`."""

    notes: list[Note]


class MelodyExtractionResult(BaseModel):
    """melody_handler's job result - see KickDetectionResult."""

    notes: list[Note]


def build_separate_stems_job(
    *, job_id: str, object_key: str, original_filename: str, lane_count: int
) -> dict:
    """Builds the first-phase job: demucs stem separation. Added standalone (no children)
    rather than nested in the beatmap-generation flow tree, because BullMQ jobs can only have a
    single parent - kick_handler and melody_handler both need this job's output, so they can't
    be modeled as its children without running separation twice. stem_handler instead
    dynamically adds the second-phase flow (see build_generate_beatmap_flow) once separation
    finishes and the stems are in storage."""

    payload = SeparateStemsJobPayload(
        job_id=job_id, object_key=object_key, original_filename=original_filename, lane_count=lane_count
    )
    return {
        "name": SEPARATE_STEMS_JOB_NAME,
        "queueName": STEM_SEPARATION_QUEUE_NAME,
        "data": payload.model_dump(),
    }


def build_generate_beatmap_flow(
    *, job_id: str, object_key: str, original_filename: str, lane_count: int, duration: int, bpm: int
) -> dict:
    """Builds the second-phase BullMQ Flow tree, added by stem_handler once the drum/melody
    stems are already in storage (and duration/bpm already detected from the original mix): a
    root orchestration job whose processor only becomes runnable once both children - kick
    onset detection and melody extraction, which run in parallel - have completed.
    `failParentOnFailure` on the children ensures a child that exhausts its retries also fails
    the root, instead of leaving it stuck waiting forever."""

    child_opts = {"failParentOnFailure": True}
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
        "children": [
            {
                "name": DETECT_KICK_ONSETS_JOB_NAME,
                "queueName": KICK_ONSET_QUEUE_NAME,
                "data": kick_payload.model_dump(),
                "opts": child_opts,
            },
            {
                "name": EXTRACT_MELODY_JOB_NAME,
                "queueName": MELODY_EXTRACTION_QUEUE_NAME,
                "data": melody_payload.model_dump(),
                "opts": child_opts,
            },
        ],
    }


def create_flow_producer(config: Config) -> FlowProducer:
    return FlowProducer({"connection": config.redis_url})


def get_flow_producer(request: Request) -> FlowProducer:
    """FastAPI dependency: the BullMQ flow producer bound to this app instance's lifespan."""
    return request.app.state.flow_producer
