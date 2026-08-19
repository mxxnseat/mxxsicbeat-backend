import os
from pathlib import Path

import pytest

from tests.golden.tracks import FAST_MONO_PIANO, GoldenTrack

_DEFAULT_PLOT_DIR = Path(__file__).resolve().parent / "output"


@pytest.fixture(scope="session")
def fast_mono_piano() -> GoldenTrack:
    return FAST_MONO_PIANO


@pytest.fixture(scope="session")
def plot_dir() -> Path | None:
    """Where diagnostic figures go - `GOLDEN_PLOT_DIR=` (empty) turns them off."""
    configured = os.environ.get("GOLDEN_PLOT_DIR")
    if configured is None:
        return _DEFAULT_PLOT_DIR
    return Path(configured) if configured.strip() else None
