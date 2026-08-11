from dataclasses import dataclass
from pathlib import Path

import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict


@dataclass(frozen=True, slots=True)
class MelodyNoteEvent:
    start_time: float
    end_time: float
    pitch: int
    amplitude: float


@dataclass(frozen=True, slots=True)
class ExtractedMelody:
    note_events: list[MelodyNoteEvent]
    midi_path: Path


def _rescale_tempo(midi_data: pretty_midi.PrettyMIDI, bpm: float) -> pretty_midi.PrettyMIDI:
    """Re-tag a MIDI file's declared tempo without moving any notes.

    Basic Pitch always writes note on/off times as real seconds but tags the file with a
    placeholder 120 BPM - DAWs that build ticks from that tag (rather than the file's actual
    note spacing) will play it back at the wrong speed unless the project tempo happens to
    already be 120. Writing the real song tempo here keeps every note's absolute time identical
    while giving DAWs/notation apps the correct beat grid to read from.
    """
    rescaled = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=midi_data.resolution)
    rescaled.instruments = midi_data.instruments
    return rescaled


def _build_note_events(note_events: list[tuple]) -> list[MelodyNoteEvent]:
    """basic-pitch's raw tuples are (start_time, end_time, pitch, amplitude, pitch_bends); pitch
    is the note's single quantized MIDI number (pitch_bends, the sub-semitone wobble within the
    note, is dropped - lane assignment only needs one pitch per note)."""
    return [
        MelodyNoteEvent(
            start_time=start_time, end_time=end_time, pitch=int(pitch), amplitude=float(amplitude)
        )
        for start_time, end_time, pitch, amplitude, _pitch_bends in note_events
    ]


def extract_melody(melody_stem_path: Path, work_dir: Path, bpm: float) -> ExtractedMelody:
    """End-to-end pitch/MIDI extraction against the already-separated melody stem (drums and
    vocals removed upstream by stem_handler): basic-pitch transcribes note events off it, and
    the MIDI's placeholder 120bpm tag is rewritten to the track's real bpm before saving. `bpm`
    is supplied by the caller (stem_handler detects it once, from the original mix, rather than
    this re-detecting it from a stem with the rhythm section removed)."""
    _, midi_data, note_events = predict(
        str(melody_stem_path), model_or_model_path=ICASSP_2022_MODEL_PATH
    )

    midi_data = _rescale_tempo(midi_data, bpm)

    midi_path = work_dir / f"{melody_stem_path.stem}.mid"
    midi_data.write(str(midi_path))

    return ExtractedMelody(note_events=_build_note_events(note_events), midi_path=midi_path)
