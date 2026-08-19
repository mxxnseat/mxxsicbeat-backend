"""Golden fixtures: a rendered audio file paired with the MIDI it was rendered from.

The MIDI is the ground truth - every `note_on` in it is an event the onset detector
is expected to find in the audio, at the same time.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import librosa
import mido
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# note_ons closer together than this are one chord, i.e. one audible attack - no onset
# detector can ever pull them apart, so ground truth must not count them twice.
_CHORD_WINDOW_S = 0.02


def midi_onset_times(midi_path: Path, chord_window_s: float = _CHORD_WINDOW_S) -> list[float]:
    """Absolute times (seconds) of every audible attack in `midi_path`."""
    note_on_times: list[float] = []
    elapsed = 0.0
    for message in mido.MidiFile(str(midi_path)):
        elapsed += message.time
        # a note_on with velocity 0 is a note_off in disguise, not an attack
        if message.type == "note_on" and message.velocity > 0:
            note_on_times.append(elapsed)

    note_on_times.sort()
    attacks: list[float] = []
    for time in note_on_times:
        if not attacks or time - attacks[-1] > chord_window_s:
            attacks.append(time)
    return attacks


@dataclass(frozen=True, slots=True)
class GoldenTrack:
    name: str

    @property
    def audio_path(self) -> Path:
        return ASSETS_DIR / self.name / "song.wav"

    @property
    def midi_path(self) -> Path:
        return ASSETS_DIR / self.name / "song.mid"

    @property
    def expected_onsets(self) -> list[float]:
        return midi_onset_times(self.midi_path)

    def load_audio(self) -> tuple[np.ndarray, int]:
        signal, sample_rate = _load_audio(self.audio_path)
        return signal, int(sample_rate)


@cache
def _load_audio(audio_path: Path) -> tuple[np.ndarray, float]:
    return librosa.load(audio_path, sr=None, mono=True)


FAST_MONO_PIANO = GoldenTrack("fast-mono-piano")
