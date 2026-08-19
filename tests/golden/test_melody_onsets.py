import numpy as np
import pytest

from app.domains.maps.services.notes_service import build_melody_notes
from tests.golden.matching import compare_onsets, deviations_ms
from tests.golden.plots import render_spectral_flux

pytestmark = pytest.mark.golden

_MAX_ABS_DEVIATION_MS = 25.0


@pytest.fixture(scope="module")
def piano_onsets(fast_mono_piano) -> list[float]:
    signal, sample_rate = fast_mono_piano.load_audio()
    notes = build_melody_notes(signal, sample_rate, 120, 2)
    return [note.time / 1000 for note in notes]


@pytest.fixture(scope="module", autouse=True)
def spectral_flux_plot(fast_mono_piano, piano_onsets, plot_dir):
    """Render the flux/threshold/deviation figure alongside the assertions below."""
    if plot_dir is None:
        return None
    path = render_spectral_flux(fast_mono_piano, piano_onsets, plot_dir, _MAX_ABS_DEVIATION_MS)
    print(f"\ngolden plot: {path}")
    return path


def test_ground_truth_is_readable(fast_mono_piano):
    expected = fast_mono_piano.expected_onsets

    assert expected == sorted(expected)
    assert len(expected) == 72


def test_detects_one_onset_per_midi_note(fast_mono_piano, piano_onsets):
    expected = fast_mono_piano.expected_onsets

    assert len(piano_onsets) == len(expected)


def test_every_midi_note_is_matched_by_a_detection(fast_mono_piano, piano_onsets):
    comparison = compare_onsets(fast_mono_piano.expected_onsets, piano_onsets)

    assert (len(comparison.missed), len(comparison.spurious)) == (0, 0), comparison.summary()


def test_detected_onsets_land_close_to_the_midi_note(fast_mono_piano, piano_onsets):
    comparison = compare_onsets(fast_mono_piano.expected_onsets, piano_onsets)

    assert np.max(np.abs(deviations_ms(comparison))) <= _MAX_ABS_DEVIATION_MS, comparison.summary()
