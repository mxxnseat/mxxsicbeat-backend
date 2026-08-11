from pathlib import Path

import librosa
import numpy as np

from app.domains.maps.dsp.kick_onseter import KickOnseter

# Demucs' "drums" stem is never truly silent on a drum-less track - it's a learned mask, not a
# drum-presence classifier, so it still carries low-level bleed (bass transients, percussive
# plucks, reverb tails). librosa's onset_detect works off *relative* spectral-flux peaks with
# no absolute-energy floor, so that bleed alone reliably gets flagged as onsets. Gating on how
# loud the drum stem is relative to the original mix catches this at the source, rather than
# trying to out-tune the onset detector's sensitivity. 0.1 was picked as a conservative floor -
# a mix with real drums carries much more of the overall energy than that.
_MIN_DRUM_TO_MIX_RMS_RATIO = 0.1


def _rms(audio_path: Path) -> float:
    signal, _ = librosa.load(audio_path, sr=None, mono=True)
    return float(np.sqrt(np.mean(np.square(signal)))) if len(signal) else 0.0


def has_significant_drum_energy(drum_stem_path: Path, original_path: Path) -> bool:
    """Whether the separated drum stem carries enough energy relative to the original mix to
    plausibly contain real drum hits, as opposed to demucs separation leakage/artifacts."""
    mix_rms = _rms(original_path)
    if mix_rms <= 0:
        return False
    return (_rms(drum_stem_path) / mix_rms) >= _MIN_DRUM_TO_MIX_RMS_RATIO


def detect_kick_onsets(drums_path: Path) -> tuple[list[float], KickOnseter]:
    drums_signal, sample_rate = librosa.load(drums_path, sr=None, mono=True)
    onseter = KickOnseter(sample_rate=int(sample_rate))
    onset_times = onseter.detect_onsets(drums_signal)
    return onset_times, onseter
