from app.common.exceptions import (
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)


class MapJobNotFoundError(NotFoundError):
    code = "map_job_not_found"


class BeatmapNotFoundError(NotFoundError):
    code = "beatmap_not_found"


class UnsupportedAudioTypeError(UnsupportedMediaTypeError):
    code = "unsupported_audio_type"


class InvalidLaneCountError(ValidationError):
    code = "invalid_lane_count"


class AudioTooLargeError(PayloadTooLargeError):
    code = "audio_too_large"
