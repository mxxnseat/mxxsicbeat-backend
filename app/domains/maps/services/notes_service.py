from functools import lru_cache

import librosa
import numpy as np
import scipy.ndimage

from app.domains.maps.dsp.kick_onseter import KickOnseter
from app.domains.maps.dtos.notes import Note, NoteType

_HOP_LENGTH = 512
_N_FFT = 2048
_MS_PER_MINUTE = 60_000

# SuperFlux (Böck & Widmer, DAFx-13), equation numbers below refer to the paper:
# https://www.dafx.de/paper-archive/2013/papers/09.dafx2013_submission_12.pdf
_SUPERFLUX_BANDS_PER_OCTAVE = 24
_SUPERFLUX_FMIN = 27.5
_SUPERFLUX_FMAX = 16000.0
_SUPERFLUX_MAX_FILTER_BANDS = 3
_SUPERFLUX_WINDOW_RATIO = 0.5

_ONSET_BASELINE_WINDOW_S = 0.5
_ONSET_Z_THRESH = 2.5
# Section 2.3's combination_width: two attacks closer together than this are one onset.
_ONSET_WAIT_S = 0.03

# Level every stem is scaled to before the novelty is measured, so that the absolute
# threshold below means the same thing on a quiet track as on a loud one. The reference is
# a high quantile rather than the peak: one click or pop would otherwise become the ruler
# and shrink every real onset underneath the threshold. Stems quieter than the silence
# level are treated as empty rather than amplified into their own noise floor.
_ONSET_NORMALIZE_QUANTILE = 0.999
_ONSET_SILENCE_LEVEL = 1e-4

# The floor under the local threshold, in units of `_superflux`'s summed log10 rise over
# the filterbank - so it is only meaningful on normalized audio, and has to be re-derived
# if the filterbank or n_fft changes. Chosen at the geometric center of the range that
# scored perfectly on the golden track, leaving ~2.5x of room on either side before either
# decay ripple gets in or quiet notes drop out.
_ONSET_MIN_NOVELTY = 3.0

_OFFSET_DECAY_DB = 15.0
_OFFSET_MIN_DUR_S = 0.05
_OFFSET_MAX_DUR_S = 4.0
_OFFSET_N_MELS = 64
_OFFSET_FMAX = 8000.0
_OFFSET_BAND_COUNT = 5
_OFFSET_BAND_WINDOW = 3
_OFFSET_BAND_FLOOR_DB = 60.0


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


@lru_cache(maxsize=4)
def _quarter_tone_filterbank(sr: int, n_fft: int) -> np.ndarray:
    """F(k, m) of equation 4: triangular filters a quarter-tone apart, 27.5...16000 Hz.

    On a logarithmic frequency scale a semitone is always the same number of bands wide, so
    the max filter below searches a constant range whatever the pitch. Filters are left
    unnormalized (section 2.1 ii), and ones whose centers collapse onto the same FFT bin -
    everything under ~735 Hz at 44.1 kHz, where a quarter-tone is narrower than a bin - are
    dropped instead of duplicated.
    """
    fmax = min(_SUPERFLUX_FMAX, sr / 2)
    band_count = int(np.floor(np.log2(fmax / _SUPERFLUX_FMIN) * _SUPERFLUX_BANDS_PER_OCTAVE)) + 1
    centers = _SUPERFLUX_FMIN * 2.0 ** (np.arange(band_count) / _SUPERFLUX_BANDS_PER_OCTAVE)

    bins = np.unique(np.round(centers * n_fft / sr).astype(int))
    n_bins = n_fft // 2 + 1
    bins = bins[(bins > 0) & (bins < n_bins - 1)]
    if len(bins) < 3:
        return np.zeros((n_bins, 0))

    filterbank = np.zeros((n_bins, len(bins) - 2))
    for band, (left, center, right) in enumerate(
        zip(bins[:-2], bins[1:-1], bins[2:], strict=True)
    ):
        filterbank[left : center + 1, band] = np.linspace(0.0, 1.0, center - left + 1)
        filterbank[center : right + 1, band] = np.linspace(1.0, 0.0, right - center + 1)
    return filterbank


def _frame_lag(n_fft: int, hop_length: int) -> int:
    """mu of equation 2: how many frames back the difference reaches.

    Neighbouring frames overlap so heavily that their difference is mostly noise, so the
    comparison is made against a frame far enough back that the two windows barely overlap.
    """
    window = librosa.filters.get_window("hann", n_fft)
    above_ratio = np.flatnonzero(window > _SUPERFLUX_WINDOW_RATIO)
    if len(above_ratio) == 0:
        return 1
    return max(1, int(np.floor((n_fft / 2 - above_ratio[0]) / hop_length + 0.5)))


def _superflux(S: np.ndarray, sr: int, hop_length: int = _HOP_LENGTH) -> np.ndarray:
    """SF* of equation 6 - spectral flux with maximum-filter trajectory tracking.

    Drop-in replacement for `_spectral_flux`: same magnitude spectrogram in, same per-frame
    novelty out. The difference is taken against a frequency-widened copy of an earlier
    frame, so energy that only wobbles in frequency - vibrato, or the drifting partials of a
    decaying string - falls inside the widened band and cancels instead of reading as a new
    onset.
    """
    n_fft = 2 * (S.shape[0] - 1)
    filterbank = _quarter_tone_filterbank(sr, n_fft)
    if filterbank.shape[1] == 0:
        return np.zeros(S.shape[1])

    # Equation 4. Filtering first and taking the log afterwards - and adding 1 before the
    # log - keeps the curve near-linear at low magnitudes, so the ripple of a decaying tail
    # stays small instead of being blown up the way a plain dB scale blows it up.
    spectrogram = np.log10(filterbank.T @ S + 1.0)

    mu = _frame_lag(n_fft, hop_length)
    if spectrogram.shape[1] <= mu:
        return np.zeros(S.shape[1])

    # Equation 5: widen each band over its direct neighbours, within that frame only.
    maximum_filtered = scipy.ndimage.maximum_filter1d(
        spectrogram, size=_SUPERFLUX_MAX_FILTER_BANDS, axis=0, mode="nearest"
    )
    difference = spectrogram[:, mu:] - maximum_filtered[:, :-mu]
    flux = np.sum(np.maximum(difference, 0.0), axis=0)
    return np.concatenate([np.zeros(mu), flux])


def _normalize_level(y: np.ndarray) -> np.ndarray | None:
    reference = float(np.quantile(np.abs(y), _ONSET_NORMALIZE_QUANTILE))
    if reference < _ONSET_SILENCE_LEVEL:
        return None
    return y / reference


def _onset_threshold(novelty: np.ndarray, sr: int, hop_length: int = _HOP_LENGTH) -> np.ndarray:
    win = max(3, int(round(_ONSET_BASELINE_WINDOW_S * sr / hop_length)))
    baseline = scipy.ndimage.median_filter(novelty, size=win, mode="nearest")
    mad = scipy.ndimage.median_filter(np.abs(novelty - baseline), size=win, mode="nearest")
    local_std = mad / 0.6745 + 1e-6
    # The local threshold follows the novelty down into quiet passages, so the ripple of a
    # decaying tail still clears it - it only ever asks an onset to stand out from its
    # neighbours. The absolute floor is the part a decaying tail cannot reach.
    return np.maximum(baseline + _ONSET_Z_THRESH * local_std, _ONSET_MIN_NOVELTY)

def _pick_onsets(novelty: np.ndarray, sr: int, hop_length: int = _HOP_LENGTH) -> np.ndarray:
    threshold = _onset_threshold(novelty, sr, hop_length)
    residual = np.maximum(novelty - threshold, 0.0)

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


def _select_bands(mel_db: np.ndarray, start_frame: int) -> np.ndarray:
    n_frames = mel_db.shape[1]
    post = mel_db[:, start_frame : min(start_frame + _OFFSET_BAND_WINDOW, n_frames)].mean(axis=1)

    pre_start = max(start_frame - _OFFSET_BAND_WINDOW, 0)
    if pre_start < start_frame:
        rise = post - mel_db[:, pre_start:start_frame].mean(axis=1)
    else:
        rise = post

    audible = post >= post.max() - _OFFSET_BAND_FLOOR_DB
    ranked = np.where(audible, rise, -np.inf)
    return np.argsort(ranked)[-_OFFSET_BAND_COUNT:]


def _estimate_offsets(
    S: np.ndarray, sr: int, onset_times: np.ndarray, hop_length: int = _HOP_LENGTH
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=_OFFSET_N_MELS, fmax=_OFFSET_FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=hop_length)

    n = len(onset_times)
    offsets = np.empty(n)
    for i, t_on in enumerate(onset_times):
        start_frame = min(np.searchsorted(times, t_on), len(times) - 1)
        next_onset = onset_times[i + 1] if i + 1 < n else times[-1]
        hard_cap = min(t_on + _OFFSET_MAX_DUR_S, next_onset, times[-1])
        cap_frame = min(np.searchsorted(times, hard_cap), len(times) - 1)

        bands = _select_bands(mel_db, start_frame)
        profile = mel_db[bands, start_frame : max(cap_frame, start_frame + 1)].mean(axis=0)
        decayed = np.flatnonzero(profile < profile[0] - _OFFSET_DECAY_DB)

        if len(decayed) == 0:
            end_time = hard_cap
        else:
            end_time = max(times[start_frame + decayed[0]], t_on + _OFFSET_MIN_DUR_S)
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
    beats = duration // bpm
    return max(1, int(beats))

def build_melody_notes(y: np.ndarray, sr: int, bpm: int, lane_count: int) -> list[Note]:
    signal = _normalize_level(y)
    if signal is None:
        return []

    S = np.abs(librosa.stft(signal, n_fft=_N_FFT, hop_length=_HOP_LENGTH))
    flux = _superflux(S, sr, hop_length=_HOP_LENGTH)
    onset_times = _pick_onsets(flux, sr, hop_length=_HOP_LENGTH)
    if len(onset_times) == 0:
        return []

    offsets = _estimate_offsets(S, sr, onset_times, hop_length=_HOP_LENGTH)
    lanes = _spectral_centroid_lanes(
        signal, sr, onset_times, offsets, lane_count, hop_length=_HOP_LENGTH
    )
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