import numpy as np

from app.domains.maps.dsp.kick_onseter import KickOnseter


def _onseter_with_odf(odf: list[float], hop_length: int = 512, sample_rate: int = 44100) -> KickOnseter:
    onseter = KickOnseter(sample_rate=sample_rate, hop_length=hop_length)
    onseter.odf = np.array(odf)
    onseter.frame_rate = sample_rate / hop_length
    return onseter


def test_strength_at_normalizes_against_peak():
    onseter = _onseter_with_odf([0.0, 2.0, 1.0])

    assert onseter.strength_at(0.0) == 0.0
    assert onseter.strength_at(onseter.frame_rate**-1) == 1.0


def test_strength_at_returns_zero_before_onsets_detected():
    onseter = KickOnseter()

    assert onseter.strength_at(0.0) == 0.0
