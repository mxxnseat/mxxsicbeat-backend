import librosa
import numpy as np
import scipy.ndimage

from app.domains.maps.dsp.kick_onseter import KickOnseter
from app.domains.maps.dtos.notes import Note, NoteType

_HOP_LENGTH = 512
_N_FFT = 2048

_ONSET_BASELINE_WINDOW_S = 0.5
_ONSET_Z_THRESH = 2.5
_ONSET_WAIT_S = 0.1

_OFFSET_DECAY_DB = 20.0
_OFFSET_MIN_DUR_S = 0.05
_OFFSET_MAX_DUR_S = 4.0


def build_drum_notes(onset_times: list[float], onseter: KickOnseter, lane_count: int) -> list[Note]:
    return [
        Note(
            time=round(onset_time * 1000),
            lane=index % lane_count,
            energy=onseter.strength_at(onset_time),
            note_type=NoteType.DRUM,
            combo=1
        )
        for index, onset_time in enumerate(onset_times)
    ]


def _spectral_flux(S: np.ndarray) -> np.ndarray:
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    diff = np.diff(S_db, axis=1, prepend=S_db[:, :1])
    return np.sum(np.maximum(diff, 0.0), axis=0)


def _pick_onsets(novelty: np.ndarray, sr: int, hop_length: int = _HOP_LENGTH) -> np.ndarray:
    win = max(3, int(round(_ONSET_BASELINE_WINDOW_S * sr / hop_length)))
    baseline = scipy.ndimage.median_filter(novelty, size=win, mode="nearest")
    mad = scipy.ndimage.median_filter(np.abs(novelty - baseline), size=win, mode="nearest")
    local_std = mad / 0.6745 + 1e-6
    residual = np.maximum(novelty - baseline - _ONSET_Z_THRESH * local_std, 0.0)

    wait_frames = int(round(_ONSET_WAIT_S * sr / hop_length))
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=residual,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        normalize=False,
        delta=1e-6,
        wait=wait_frames,
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def _estimate_offsets(
    S: np.ndarray, sr: int, onset_times: np.ndarray, hop_length: int = _HOP_LENGTH
) -> np.ndarray:
    energy_db = librosa.power_to_db(np.sum(S**2, axis=0), ref=np.max)
    times = librosa.frames_to_time(np.arange(len(energy_db)), sr=sr, hop_length=hop_length)

    n = len(onset_times)
    offsets = np.empty(n)
    for i, t_on in enumerate(onset_times):
        start_frame = min(np.searchsorted(times, t_on), len(energy_db) - 1)
        peak_db = energy_db[start_frame]
        threshold_db = peak_db - _OFFSET_DECAY_DB
        next_onset = onset_times[i + 1] if i + 1 < n else times[-1]
        hard_cap = min(t_on + _OFFSET_MAX_DUR_S, next_onset, times[-1])

        end_time = hard_cap
        for f in range(start_frame, len(energy_db)):
            t = times[f]
            if t >= hard_cap:
                break
            if energy_db[f] < threshold_db:
                end_time = max(t, t_on + _OFFSET_MIN_DUR_S)
                break
        offsets[i] = min(end_time, hard_cap)
    return offsets


def _onset_strengths(
    flux: np.ndarray, sr: int, onset_times: np.ndarray, hop_length: int = _HOP_LENGTH
) -> np.ndarray:
    if len(flux) == 0 or len(onset_times) == 0:
        return np.zeros(len(onset_times))
    peak = float(np.max(flux))
    if peak <= 0:
        return np.zeros(len(onset_times))
    frames = np.clip(np.round(onset_times * sr / hop_length).astype(int), 0, len(flux) - 1)
    return np.clip(flux[frames] / peak, 0.0, 1.0)


def _spectral_centroid_lanes(
    y: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    offsets: np.ndarray,
    lane_count: int,
    hop_length: int = _HOP_LENGTH,
) -> np.ndarray:
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop_length)

    note_centroids = np.empty(len(onset_times))
    for i, (t_on, t_off) in enumerate(zip(onset_times, offsets, strict=True)):
        mask = (times >= t_on) & (times < max(t_off, t_on + 0.05))
        if np.any(mask):
            note_centroids[i] = np.mean(cent[mask])
        else:
            idx = min(np.searchsorted(times, t_on), len(cent) - 1)
            note_centroids[i] = cent[idx]

    quantile_edges = np.quantile(note_centroids, np.linspace(0, 1, lane_count + 1))
    quantile_edges[0] -= 1.0
    return np.digitize(note_centroids, quantile_edges[1:-1])

def calculate_melody_combo(duration: int, bpm: int) -> int:
    beats = (duration) // (bpm)
    return max(1, int(beats))

def build_melody_notes(y: np.ndarray, sr: int, bpm: int, lane_count: int) -> list[Note]:
    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP_LENGTH))
    flux = _spectral_flux(S)
    onset_times = _pick_onsets(flux, sr, hop_length=_HOP_LENGTH)
    if len(onset_times) == 0:
        return []

    offsets = _estimate_offsets(S, sr, onset_times, hop_length=_HOP_LENGTH)
    lanes = _spectral_centroid_lanes(y, sr, onset_times, offsets, lane_count, hop_length=_HOP_LENGTH)
    energies = _onset_strengths(flux, sr, onset_times, hop_length=_HOP_LENGTH)

    return [
        build_melody_note(onset_time, offset, lane, energy, bpm)
        for onset_time, offset, lane, energy in zip(onset_times, offsets, lanes, energies, strict=True)
    ]

def build_melody_note(onset_time: float, offset: float, lane: float, energy: float, bpm: int) -> Note:
    duration = round((offset - onset_time) * 1000)
    return Note(
            time=round(onset_time * 1000),
            duration=duration,
            lane=int(lane),
            energy=float(energy),
            note_type=NoteType.MELODY,
            combo=calculate_melody_combo(duration, bpm)
        )