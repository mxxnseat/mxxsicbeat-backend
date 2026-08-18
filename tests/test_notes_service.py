import librosa
import numpy as np
import pytest

from app.domains.maps.dsp.kick_onseter import KickOnseter
from app.domains.maps.dtos.notes import NoteType
from app.domains.maps.services.notes_service import (
    _estimate_offsets,
    _onset_strengths,
    _pick_onsets,
    _spectral_centroid_lanes,
    _spectral_flux,
    build_drum_notes,
    build_melody_notes,
    calculate_melody_combo,
)

_SR = 22050
_HOP_LENGTH = 512


def _onseter_with_odf(odf: list[float], hop_length: int = 512, sample_rate: int = 44100) -> KickOnseter:
    onseter = KickOnseter(sample_rate=sample_rate, hop_length=hop_length)
    onseter.odf = np.array(odf)
    onseter.frame_rate = sample_rate / hop_length
    return onseter


def _sine(freq: float, duration: float, sr: int = _SR, amplitude: float = 0.8) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def _tone(
    freq: float, duration: float, sr: int = _SR, amplitude: float = 0.8, fade: float = 0.01
) -> np.ndarray:
    signal = _sine(freq, duration, sr=sr, amplitude=amplitude)
    fade_samples = int(sr * fade)
    envelope = np.ones_like(signal)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return signal * envelope


def test_build_drum_notes_assigns_lanes_round_robin():
    onseter = _onseter_with_odf([1.0, 1.0, 1.0, 1.0])
    notes = build_drum_notes([0.0, 0.5, 1.0, 1.5], onseter, lane_count=2)

    assert [note.lane for note in notes] == [0, 1, 0, 1]


def test_build_drum_notes_converts_seconds_to_milliseconds():
    onseter = _onseter_with_odf([1.0])
    notes = build_drum_notes([1.234], onseter, lane_count=2)

    assert notes[0].time == 1234


def test_build_drum_notes_single_lane_puts_everything_on_lane_zero():
    onseter = _onseter_with_odf([1.0, 1.0, 1.0])
    notes = build_drum_notes([0.0, 0.1, 0.2], onseter, lane_count=1)

    assert all(note.lane == 0 for note in notes)


def test_build_drum_notes_have_no_duration_and_drum_type():
    onseter = _onseter_with_odf([1.0])
    notes = build_drum_notes([0.0], onseter, lane_count=1)

    assert notes[0].duration is None
    assert notes[0].note_type == NoteType.DRUM


def test_build_drum_notes_rounds_down_below_half():
    onseter = _onseter_with_odf([1.0])
    notes = build_drum_notes([1.2344], onseter, lane_count=1)

    assert notes[0].time == 1234


def test_build_drum_notes_rounds_up_above_half():
    onseter = _onseter_with_odf([1.0])
    notes = build_drum_notes([1.2346], onseter, lane_count=1)

    assert notes[0].time == 1235


def test_build_drum_notes_rounds_exact_half_to_even():
    onseter = _onseter_with_odf([1.0, 1.0])
    notes = build_drum_notes([1.2345, 1.2335], onseter, lane_count=1)

    assert [note.time for note in notes] == [1234, 1234]


def test_spectral_flux_is_zero_when_energy_is_constant():
    S = np.full((4, 5), 2.0)

    flux = _spectral_flux(S)

    assert np.allclose(flux, 0.0)


def test_spectral_flux_spikes_on_energy_jump():
    S = np.ones((4, 6))
    S[:, 3] = 20.0

    flux = _spectral_flux(S)

    assert flux[0] == 0.0
    assert flux[3] > 0.0
    assert flux[3] > flux[1]
    assert flux[4] == 0.0


def test_pick_onsets_detects_impulses_above_local_noise_floor():
    novelty = np.zeros(200)
    novelty[50] = 5.0
    novelty[120] = 5.0

    onset_times = _pick_onsets(novelty, _SR, hop_length=_HOP_LENGTH)

    expected = librosa.frames_to_time([50, 120], sr=_SR, hop_length=_HOP_LENGTH)
    assert np.allclose(onset_times, expected)


def test_pick_onsets_ignores_flat_novelty():
    novelty = np.zeros(200)

    onset_times = _pick_onsets(novelty, _SR, hop_length=_HOP_LENGTH)

    assert len(onset_times) == 0


_N_BINS = 1 + 2048 // 2


def test_estimate_offsets_ends_when_energy_decays_below_threshold():
    S = np.full((_N_BINS, 40), 5.0)
    S[:, 10:] = 0.01

    offsets = _estimate_offsets(S, _SR, np.array([0.0]), hop_length=_HOP_LENGTH)

    expected = librosa.frames_to_time(10, sr=_SR, hop_length=_HOP_LENGTH)
    assert offsets[0] == pytest.approx(expected, abs=0.05)


def test_estimate_offsets_caps_at_next_onset():
    S = np.full((_N_BINS, 100), 5.0)

    onset_times = np.array([0.0, 0.3])
    offsets = _estimate_offsets(S, _SR, onset_times, hop_length=_HOP_LENGTH)

    assert offsets[0] == pytest.approx(0.3, abs=0.02)


def test_estimate_offsets_follows_onset_bands_through_a_sustained_bed():
    pad = _sine(180.0, 1.0)
    note = np.concatenate([_tone(3000.0, 0.2), np.zeros(int(_SR * 0.8))])
    S = np.abs(librosa.stft(pad + note, n_fft=2048, hop_length=_HOP_LENGTH))

    offsets = _estimate_offsets(S, _SR, np.array([0.0]), hop_length=_HOP_LENGTH)

    assert offsets[0] == pytest.approx(0.2, abs=0.08)


def test_onset_strengths_normalizes_against_peak_flux():
    flux = np.array([0.0, 2.0, 4.0, 1.0])
    onset_times = librosa.frames_to_time([1, 2], sr=_SR, hop_length=_HOP_LENGTH)

    strengths = _onset_strengths(flux, _SR, onset_times, hop_length=_HOP_LENGTH)

    assert strengths[0] == pytest.approx(0.5)
    assert strengths[1] == pytest.approx(1.0)


def test_onset_strengths_returns_zero_for_no_onsets():
    strengths = _onset_strengths(np.array([1.0, 2.0]), _SR, np.array([]), hop_length=_HOP_LENGTH)

    assert len(strengths) == 0


def test_spectral_centroid_lanes_separates_low_and_high_tones():
    low = _sine(220.0, 0.2)
    high = _sine(2000.0, 0.2)
    y = np.concatenate([low, high])

    lanes = _spectral_centroid_lanes(
        y, _SR, onset_times=np.array([0.0, 0.2]), offsets=np.array([0.2, 0.4]), lane_count=2
    )

    assert lanes[0] == 0
    assert lanes[1] == 1


def test_spectral_centroid_lanes_single_lane_puts_everything_on_lane_zero():
    low = _sine(220.0, 0.2)
    high = _sine(2000.0, 0.2)
    y = np.concatenate([low, high])

    lanes = _spectral_centroid_lanes(
        y, _SR, onset_times=np.array([0.0, 0.2]), offsets=np.array([0.2, 0.4]), lane_count=1
    )

    assert list(lanes) == [0, 0]


def test_build_melody_notes_detects_notes_in_different_lanes():
    silence = np.zeros(int(_SR * 0.1))
    low = _tone(220.0, 0.3)
    high = _tone(1760.0, 0.3)
    y = np.concatenate([low, silence, high, np.zeros(int(_SR * 0.05))])

    notes = build_melody_notes(y, _SR, 120, lane_count=2)

    assert len(notes) >= 2
    assert all(note.note_type == NoteType.MELODY for note in notes)
    assert all(0.0 <= note.energy <= 1.0 for note in notes)
    assert all(note.duration is not None and note.duration >= 0 for note in notes)
    assert all(note.lane == (0 if note.time < 350 else 1) for note in notes)
    assert any(note.lane == 0 for note in notes)
    assert any(note.lane == 1 for note in notes)


def test_build_melody_notes_returns_empty_list_for_silence():
    y = np.zeros(_SR)

    assert build_melody_notes(y, _SR, 120, lane_count=2) == []


def test_calculate_melody_combo_counts_beats_spanned_by_the_hold():
    assert calculate_melody_combo(duration=2000, bpm=120) == 4
    assert calculate_melody_combo(duration=500, bpm=120) == 1


def test_calculate_melody_combo_floors_short_holds_at_one():
    assert calculate_melody_combo(duration=60, bpm=128) == 1
