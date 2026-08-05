import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


class KickOnseter:
    """Kick-drum onset detector: isolate the kick's frequency range with a
    low-pass filter, then hand off to librosa's onset detector.

    Several iterations of a hand-rolled adaptive-threshold detector (STFT
    spectral flux, then a time-domain envelope follower) kept trading one
    failure mode for another across different tempos - a fix that made a
    fast blast-beat section detectable badly over-triggered on a slower
    track (shoulder-bumps on a kick's own decay, snare/hihat bleed), and
    vice versa. librosa's onset_detect is a mature, widely-validated
    implementation; band-limiting to the kick's frequency range first still
    does the job of separating kick from the rest of the kit within the
    isolated drum stem.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        cutoff_hz: float = 300.0,
        hop_length: int = 512,
        backtrack: bool = False,
    ):
        self.sample_rate = sample_rate
        self.cutoff_hz = cutoff_hz
        self.hop_length = hop_length
        self.backtrack = backtrack

        self.odf: np.ndarray | None = None
        self.frame_rate: float | None = None

    def _isolate_kick_band(self, audio_signal: np.ndarray) -> np.ndarray:
        sos = butter(4, self.cutoff_hz, btype="low", fs=self.sample_rate, output="sos")
        return sosfiltfilt(sos, audio_signal)

    def detect_onsets(self, audio_signal: np.ndarray) -> list[float]:
        kick_band_signal = self._isolate_kick_band(audio_signal)

        self.odf = librosa.onset.onset_strength(
            y=kick_band_signal, sr=self.sample_rate, hop_length=self.hop_length
        )
        self.frame_rate = self.sample_rate / self.hop_length

        onset_frames = librosa.onset.onset_detect(
            onset_envelope=self.odf,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            units="frames",
        )
        return list(librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=self.hop_length))

    def strength_at(self, onset_time: float) -> float:
        """Normalized (0-1) onset detection function value at the given time, for use as a
        per-note energy hint. Returns 0.0 if onsets haven't been detected yet or the ODF is flat."""
        if self.odf is None or self.frame_rate is None or len(self.odf) == 0:
            return 0.0
        frame = int(round(onset_time * self.frame_rate))
        frame = max(0, min(frame, len(self.odf) - 1))
        peak = float(np.max(self.odf))
        if peak <= 0:
            return 0.0
        return float(np.clip(self.odf[frame] / peak, 0.0, 1.0))
