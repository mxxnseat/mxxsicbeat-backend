"""Aligning detected onsets against the ground-truth onsets of a golden track."""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_TOLERANCE_S = 0.05


@dataclass(frozen=True, slots=True)
class OnsetMatch:
    """One detected onset paired with the expected onset it accounts for."""

    expected: float
    detected: float

    @property
    def deviation(self) -> float:
        return self.detected - self.expected


@dataclass(frozen=True, slots=True)
class OnsetComparison:
    matches: list[OnsetMatch] = field(default_factory=list)
    missed: list[float] = field(default_factory=list)
    spurious: list[float] = field(default_factory=list)

    @property
    def expected_count(self) -> int:
        return len(self.matches) + len(self.missed)

    @property
    def detected_count(self) -> int:
        return len(self.matches) + len(self.spurious)

    @property
    def recall(self) -> float:
        return len(self.matches) / self.expected_count if self.expected_count else 1.0

    @property
    def precision(self) -> float:
        return len(self.matches) / self.detected_count if self.detected_count else 1.0

    @property
    def max_abs_deviation(self) -> float:
        return max((abs(match.deviation) for match in self.matches), default=0.0)

    def summary(self) -> str:
        return (
            f"expected={self.expected_count} detected={self.detected_count} "
            f"matched={len(self.matches)} missed={len(self.missed)} spurious={len(self.spurious)} "
            f"precision={self.precision:.2f} recall={self.recall:.2f}\n"
            f"  missed:   {[round(t, 3) for t in self.missed]}\n"
            f"  spurious: {[round(t, 3) for t in self.spurious]}"
        )


def compare_onsets(
    expected: list[float], detected: list[float], tolerance_s: float = DEFAULT_TOLERANCE_S
) -> OnsetComparison:
    """Pair each expected onset with the nearest unclaimed detection within `tolerance_s`.

    Pairs are taken in ascending order of distance, so a detection sitting between two
    expected onsets is credited to the one it is actually closest to.
    """
    candidates = sorted(
        (
            (abs(det - exp), exp_index, det_index)
            for exp_index, exp in enumerate(expected)
            for det_index, det in enumerate(detected)
            if abs(det - exp) <= tolerance_s
        )
    )

    claimed_expected: set[int] = set()
    claimed_detected: set[int] = set()
    matches: list[OnsetMatch] = []
    for _, exp_index, det_index in candidates:
        if exp_index in claimed_expected or det_index in claimed_detected:
            continue
        claimed_expected.add(exp_index)
        claimed_detected.add(det_index)
        matches.append(OnsetMatch(expected=expected[exp_index], detected=detected[det_index]))

    matches.sort(key=lambda match: match.expected)
    return OnsetComparison(
        matches=matches,
        missed=[time for index, time in enumerate(expected) if index not in claimed_expected],
        spurious=[time for index, time in enumerate(detected) if index not in claimed_detected],
    )


def deviations_ms(comparison: OnsetComparison) -> np.ndarray:
    return np.array([match.deviation for match in comparison.matches]) * 1000
