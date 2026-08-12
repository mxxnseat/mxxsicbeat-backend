import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


class KickOnseter:
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
        if self.odf is None or self.frame_rate is None or len(self.odf) == 0:
            return 0.0
        frame = int(round(onset_time * self.frame_rate))
        frame = max(0, min(frame, len(self.odf) - 1))
        peak = float(np.max(self.odf))
        if peak <= 0:
            return 0.0
        return float(np.clip(self.odf[frame] / peak, 0.0, 1.0))
