"""Diagnostic matplotlib output for the golden onset tests.

The golden assertions tell you *that* detection drifted; these figures tell you where.
One PNG per track, three stacked panels sharing a time axis:

1. the spectral flux novelty curve against the adaptive threshold it has to clear,
2. the residual the peak picker actually sees, with every detection stemmed,
3. the signed deviation of each matched detection from its MIDI note.

The threshold here is recomputed from the tunables `notes_service` exports rather than
read out of it - the detector deliberately keeps that math private, so the constants are
the shared part.
"""

from pathlib import Path

import librosa
import matplotlib
import numpy as np
import scipy.ndimage

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from app.domains.maps.services.notes_service import (  # noqa: E402
    _HOP_LENGTH,
    _N_FFT,
    _onset_threshold,
    _spectral_flux,
)
from tests.golden.matching import DEFAULT_TOLERANCE_S, OnsetComparison, compare_onsets  # noqa: E402
from tests.golden.tracks import GoldenTrack  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"
FLUX = "#2a78d6"
SPURIOUS = "#eb6834"
MISSED = "#e34948"
EXPECTED = "#8f8e89"


def _style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(GRID)


def _plot_novelty(
    ax: plt.Axes,
    times: np.ndarray,
    flux: np.ndarray,
    threshold: np.ndarray,
    comparison: OnsetComparison,
) -> None:
    ax.plot(times, flux, color=FLUX, linewidth=1.2, label="spectral flux")
    ax.plot(times, threshold, color=INK_MUTED, linewidth=1.0, linestyle="--", label="threshold")

    # dashed vs solid, so the two failure kinds stay apart without relying on hue alone
    for index, time in enumerate(comparison.missed):
        ax.axvline(
            time,
            color=MISSED,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=0.85,
            label="missed" if index == 0 else None,
        )
    for index, time in enumerate(comparison.spurious):
        ax.axvline(
            time,
            color=SPURIOUS,
            linewidth=1.0,
            alpha=0.85,
            label="spurious" if index == 0 else None,
        )

    expected = sorted([match.expected for match in comparison.matches] + comparison.missed)
    ax.plot(
        expected,
        np.full(len(expected), 0.015),
        transform=ax.get_xaxis_transform(),
        marker="|",
        markersize=7,
        linestyle="none",
        color=EXPECTED,
        label="MIDI note",
    )
    _style_axes(ax, "flux (dB/frame)")
    _legend(ax)


def _plot_residual(
    ax: plt.Axes, times: np.ndarray, residual: np.ndarray, detected: list[float]
) -> None:
    ax.fill_between(times, residual, color=FLUX, alpha=0.18, linewidth=0)
    ax.plot(times, residual, color=FLUX, linewidth=1.0)

    ax.plot(
        detected,
        np.full(len(detected), 0.97),
        transform=ax.get_xaxis_transform(),
        marker="v",
        markersize=4,
        linestyle="none",
        color=INK,
        alpha=0.7,
        label="picked onset",
    )
    _style_axes(ax, "residual (flux above threshold)")
    ax.set_ylim(bottom=0)
    _legend(ax, loc="lower right")


def _plot_deviations(ax: plt.Axes, comparison: OnsetComparison, max_abs_deviation_ms: float) -> None:
    tolerance_ms = DEFAULT_TOLERANCE_S * 1000
    ax.axhspan(-tolerance_ms, tolerance_ms, color=GRID, alpha=0.55, linewidth=0, label="match window")
    for limit in (-max_abs_deviation_ms, max_abs_deviation_ms):
        ax.axhline(limit, color=MISSED, linewidth=1.0, linestyle="--")
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)

    expected = [match.expected for match in comparison.matches]
    deviations = [match.deviation * 1000 for match in comparison.matches]
    ax.vlines(expected, 0, deviations, color=FLUX, linewidth=0.8, alpha=0.5)
    ax.plot(expected, deviations, "o", markersize=4, color=FLUX, label="detected - MIDI")

    ax.plot(
        comparison.missed,
        np.zeros(len(comparison.missed)),
        marker="x",
        markersize=6,
        linestyle="none",
        color=MISSED,
        label="missed",
    )

    _style_axes(ax, "deviation (ms)")
    ax.set_xlabel("time (s)", color=INK_MUTED, fontsize=9)
    ax.set_ylim(-tolerance_ms * 1.2, tolerance_ms * 1.2)
    _legend(ax)
    ax.annotate(
        f"limit ±{max_abs_deviation_ms:.0f} ms",
        xy=(0.995, max_abs_deviation_ms),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8,
        color=MISSED,
    )


def _legend(ax: plt.Axes, loc: str = "upper right") -> None:
    legend = ax.legend(loc=loc, fontsize=8, frameon=True, ncol=3, framealpha=1.0)
    legend.set_zorder(5)
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(INK_MUTED)


def render_spectral_flux(
    track: GoldenTrack,
    detected: list[float],
    out_dir: Path,
    max_abs_deviation_ms: float,
) -> Path:
    """Write `<track>-spectral-flux.png` into `out_dir` and return its path."""
    signal, sample_rate = track.load_audio()
    expected = track.expected_onsets
    comparison = compare_onsets(expected, detected)

    S = np.abs(librosa.stft(signal, n_fft=_N_FFT, hop_length=_HOP_LENGTH))
    flux = _spectral_flux(S)
    threshold = _onset_threshold(flux, sample_rate)
    residual = np.maximum(flux - threshold, 0.0)
    times = librosa.frames_to_time(np.arange(len(flux)), sr=sample_rate, hop_length=_HOP_LENGTH)

    width = float(np.clip(times[-1] * 0.9, 12.0, 30.0))
    figure, (novelty_ax, residual_ax, deviation_ax) = plt.subplots(
        3, 1, figsize=(width, 8.5), sharex=True, height_ratios=(3, 2, 2)
    )
    figure.patch.set_facecolor(SURFACE)

    _plot_novelty(novelty_ax, times, flux, threshold, comparison)
    _plot_residual(residual_ax, times, residual, detected)
    _plot_deviations(deviation_ax, comparison, max_abs_deviation_ms)

    novelty_ax.set_title(
        f"{track.name} - spectral flux onset detection",
        color=INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    figure.suptitle(
        f"expected {comparison.expected_count} · detected {comparison.detected_count} · "
        f"missed {len(comparison.missed)} · spurious {len(comparison.spurious)} · "
        f"precision {comparison.precision:.2f} · recall {comparison.recall:.2f} · "
        f"max |deviation| {comparison.max_abs_deviation * 1000:.1f} ms",
        color=INK_MUTED,
        fontsize=9,
        x=0.008,
        y=0.985,
        ha="left",
    )
    novelty_ax.set_xlim(0, times[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{track.name}-spectral-flux.png"
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return path
