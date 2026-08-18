from dataclasses import dataclass
from pathlib import Path

import demucs.separate


@dataclass(frozen=True, slots=True)
class SeparatedStems:
    drum_stem_path: Path
    melody_stem_path: Path


def _demucs_two_stems(two_stems: str, audio_path: Path, output_dir: Path) -> Path:
    demucs.separate.main(["--two-stems", two_stems, "-o", str(output_dir), str(audio_path)])
    return output_dir / "htdemucs" / audio_path.stem


def separate_stems(audio_path: Path, work_dir: Path) -> SeparatedStems:
    drum_stem_dir = _demucs_two_stems("drums", audio_path, work_dir)
    no_drums_path = drum_stem_dir / "no_drums.wav"

    return SeparatedStems(
        drum_stem_path=drum_stem_dir / "drums.wav",
        melody_stem_path=no_drums_path,
    )
