import numpy as np

from app.domains.maps.dsp.kick_onseter import KickOnseter
from app.domains.maps.dsp.pipeline import build_notes


def _onseter_with_odf(odf: list[float], hop_length: int = 512, sample_rate: int = 44100) -> KickOnseter:
    onseter = KickOnseter(sample_rate=sample_rate, hop_length=hop_length)
    onseter.odf = np.array(odf)
    onseter.frame_rate = sample_rate / hop_length
    return onseter


def test_build_notes_assigns_lanes_round_robin():
    onseter = _onseter_with_odf([1.0, 1.0, 1.0, 1.0])
    notes = build_notes([0.0, 0.5, 1.0, 1.5], onseter, lane_count=2)

    assert [note["lane"] for note in notes] == [0, 1, 0, 1]


def test_build_notes_converts_seconds_to_milliseconds():
    onseter = _onseter_with_odf([1.0])
    notes = build_notes([1.234], onseter, lane_count=2)

    assert notes[0]["time_ms"] == 1234


def test_build_notes_single_lane_puts_everything_on_lane_zero():
    onseter = _onseter_with_odf([1.0, 1.0, 1.0])
    notes = build_notes([0.0, 0.1, 0.2], onseter, lane_count=1)

    assert all(note["lane"] == 0 for note in notes)


def test_strength_at_normalizes_against_peak():
    onseter = _onseter_with_odf([0.0, 2.0, 1.0])

    assert onseter.strength_at(0.0) == 0.0
    assert onseter.strength_at(onseter.frame_rate**-1) == 1.0


def test_strength_at_returns_zero_before_onsets_detected():
    onseter = KickOnseter()

    assert onseter.strength_at(0.0) == 0.0
