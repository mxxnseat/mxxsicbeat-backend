from pathlib import Path

import librosa


def track_duration(audio_path: Path) -> int:
    duration_seconds = librosa.get_duration(path=str(audio_path))
    return round(duration_seconds * 1000)


def detect_bpm(audio_path: Path) -> float:
    signal, sample_rate = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sample_rate)
    return tempo.item()
