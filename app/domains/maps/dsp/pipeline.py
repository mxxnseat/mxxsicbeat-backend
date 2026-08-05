from dataclasses import dataclass
from pathlib import Path

import demucs.separate
import librosa

from app.domains.maps.dsp.kick_onseter import KickOnseter


@dataclass(frozen=True, slots=True)
class GeneratedBeatmap:
    lane_count: int
    duration_ms: int
    bpm: float | None
    notes: list[dict]
    drum_stem_path: Path
    melody_stem_path: Path


def separate_stems(audio_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Runs demucs two-stem (drums) separation, returns (drum_stem_path, melody_stem_path).
    Demucs' two-stem mode splits audio into `drums.wav` and `no_drums.wav`; the latter -
    everything except drums - stands in as the "melody" stem until a proper melodic
    separation model is wired in."""
    demucs.separate.main(["--two-stems=drums", "-o", str(output_dir), str(audio_path)])
    stem_dir = output_dir / "htdemucs" / audio_path.stem
    return stem_dir / "drums.wav", stem_dir / "no_drums.wav"


def detect_kick_onsets(drums_path: Path) -> tuple[list[float], KickOnseter]:
    drums_signal, sample_rate = librosa.load(drums_path, sr=None, mono=True)
    onseter = KickOnseter(sample_rate=sample_rate)
    onset_times = onseter.detect_onsets(drums_signal)
    return onset_times, onseter


def track_duration_ms(audio_path: Path) -> int:
    duration_seconds = librosa.get_duration(path=str(audio_path))
    return round(duration_seconds * 1000)


def build_notes(onset_times: list[float], onseter: KickOnseter, lane_count: int) -> list[dict]:
    """Turn detected onset times into lane-assigned notes.

    Lane assignment is a simple round-robin over onset order - there's no pitch or
    frequency-band signal yet to place notes more meaningfully (only kick-onset detection is
    implemented so far). Swapping this for pitch- or band-based lane assignment is the natural
    next step once basic-pitch is wired in.
    """
    return [
        {
            "time_ms": round(onset_time * 1000),
            "lane": index % lane_count,
            "energy": onseter.strength_at(onset_time),
        }
        for index, onset_time in enumerate(onset_times)
    ]


def generate_beatmap(audio_path: Path, work_dir: Path, lane_count: int) -> GeneratedBeatmap:
    """End-to-end: separate drums, detect kick onsets, assign lanes. Returns everything needed
    to build a Beatmap document except title/job_id, which the caller already knows."""
    drum_stem_path, melody_stem_path = separate_stems(audio_path, work_dir)
    onset_times, onseter = detect_kick_onsets(drum_stem_path)
    return GeneratedBeatmap(
        lane_count=lane_count,
        duration_ms=track_duration_ms(audio_path),
        bpm=None,
        notes=build_notes(onset_times, onseter, lane_count),
        drum_stem_path=drum_stem_path,
        melody_stem_path=melody_stem_path,
    )
