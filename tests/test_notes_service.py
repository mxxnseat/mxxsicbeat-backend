import numpy as np

from app.domains.maps.dsp.kick_onseter import KickOnseter
from app.domains.maps.dsp.melody_extractor import MelodyNoteEvent
from app.domains.maps.dtos.notes import NoteType
from app.domains.maps.services.notes_service import build_drum_notes, build_melody_notes


def _onseter_with_odf(odf: list[float], hop_length: int = 512, sample_rate: int = 44100) -> KickOnseter:
    onseter = KickOnseter(sample_rate=sample_rate, hop_length=hop_length)
    onseter.odf = np.array(odf)
    onseter.frame_rate = sample_rate / hop_length
    return onseter


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
    # both land exactly on a .5ms boundary (1234.5 and 1233.5) - round() rounds half-to-even,
    # so both land on 1234 rather than the naive "always round half up" result of 1235/1234.
    notes = build_drum_notes([1.2345, 1.2335], onseter, lane_count=1)

    assert [note.time for note in notes] == [1234, 1234]


def test_build_melody_notes_assigns_lanes_by_pitch_band():
    # pitches split cleanly around the median (66.5): [60, 61] below, [72, 73] above. Notes are
    # spaced 1s apart, far wider than the suppression window at 120bpm, so none get thinned.
    events = [
        MelodyNoteEvent(start_time=0.0, end_time=0.1, pitch=60, amplitude=0.5),
        MelodyNoteEvent(start_time=1.0, end_time=1.1, pitch=61, amplitude=0.5),
        MelodyNoteEvent(start_time=2.0, end_time=2.1, pitch=72, amplitude=0.5),
        MelodyNoteEvent(start_time=3.0, end_time=3.1, pitch=73, amplitude=0.5),
    ]
    notes = build_melody_notes(events, lane_count=2, bpm=120.0)

    assert [note.lane for note in notes] == [0, 0, 1, 1]


def test_build_melody_notes_keeps_simultaneous_notes_in_different_lanes():
    # a chord: two pitches far enough apart to land in different bands, at the same timestamp -
    # both should survive as a simultaneous multi-lane hit rather than being collapsed to one.
    events = [
        MelodyNoteEvent(start_time=0.5, end_time=0.6, pitch=60, amplitude=0.5),
        MelodyNoteEvent(start_time=0.5, end_time=0.6, pitch=72, amplitude=0.5),
    ]
    notes = build_melody_notes(events, lane_count=2, bpm=120.0)

    assert len(notes) == 2
    assert {note.lane for note in notes} == {0, 1}
    assert all(note.time == 500 for note in notes)


def test_build_melody_notes_suppresses_lower_salience_note_in_same_window():
    # two notes 10ms apart, well inside the ~62.5ms suppression window at 120bpm - the shorter,
    # quieter one (lower salience) should be dropped in favor of the louder, longer-held one.
    events = [
        MelodyNoteEvent(start_time=0.0, end_time=0.2, pitch=60, amplitude=0.9),
        MelodyNoteEvent(start_time=0.01, end_time=0.06, pitch=60, amplitude=0.5),
    ]
    notes = build_melody_notes(events, lane_count=1, bpm=120.0)

    assert len(notes) == 1
    assert notes[0].time == 0
    assert notes[0].energy == 0.9


def test_build_melody_notes_hysteresis_keeps_previous_lane_near_boundary():
    # boundary computed from [50, 60, 62, 75] is 61. Without hysteresis, pitch 62 (1 semitone
    # above the boundary) would band into lane 1 and then back to lane 0 for pitch 60 right
    # after it - flutter driven by noise rather than real melodic movement. With hysteresis it
    # stays on the previous note's lane (0) since it's within the margin.
    events = [
        MelodyNoteEvent(start_time=0.0, end_time=0.1, pitch=50, amplitude=0.5),
        MelodyNoteEvent(start_time=1.0, end_time=1.1, pitch=62, amplitude=0.5),
        MelodyNoteEvent(start_time=2.0, end_time=2.1, pitch=60, amplitude=0.5),
        MelodyNoteEvent(start_time=3.0, end_time=3.1, pitch=75, amplitude=0.5),
    ]
    notes = build_melody_notes(events, lane_count=2, bpm=120.0)

    assert [note.lane for note in notes] == [0, 0, 0, 1]


def test_build_melody_notes_shortens_hold_to_leave_gap_before_next_note_same_lane():
    # note1 holds until 0.5s, note2 starts at 0.55s - only 50ms of release-and-catch room, well
    # under the 200ms minimum, so note1's hold should be trimmed back to 0.35s (0.55 - 0.2).
    events = [
        MelodyNoteEvent(start_time=0.0, end_time=0.5, pitch=60, amplitude=0.5),
        MelodyNoteEvent(start_time=0.55, end_time=0.65, pitch=60, amplitude=0.5),
    ]
    notes = build_melody_notes(events, lane_count=1, bpm=120.0)

    assert notes[0].duration == 350
    assert notes[1].duration == 100


def test_build_melody_notes_hold_gap_only_applies_within_same_lane():
    # overlapping holds in different lanes are two different keys - no conflict, so neither
    # should be trimmed even though they overlap in time.
    events = [
        MelodyNoteEvent(start_time=0.0, end_time=0.5, pitch=60, amplitude=0.5),
        MelodyNoteEvent(start_time=0.1, end_time=0.4, pitch=90, amplitude=0.5),
    ]
    notes = build_melody_notes(events, lane_count=2, bpm=120.0)

    assert {note.lane for note in notes} == {0, 1}
    assert {note.duration for note in notes} == {500, 300}


def test_build_melody_notes_floors_hold_at_zero_when_onsets_too_close_for_any_gap():
    # note2 starts only 100ms after note1 - less than the 200ms minimum gap - so even trimming
    # note1's hold to zero duration can't create enough room; it should floor at 0, not go
    # negative.
    events = [
        MelodyNoteEvent(start_time=0.0, end_time=0.3, pitch=60, amplitude=0.5),
        MelodyNoteEvent(start_time=0.1, end_time=0.2, pitch=60, amplitude=0.9),
    ]
    notes = build_melody_notes(events, lane_count=1, bpm=120.0)

    assert notes[0].duration == 0


def test_build_melody_notes_converts_seconds_to_milliseconds_and_sets_duration():
    events = [MelodyNoteEvent(start_time=1.0, end_time=1.25, pitch=60, amplitude=0.5)]
    notes = build_melody_notes(events, lane_count=1, bpm=120.0)

    assert notes[0].time == 1000
    assert notes[0].duration == 250
    assert notes[0].note_type == NoteType.MELODY


def test_build_melody_notes_rounds_time_and_duration_independently():
    # start_time rounds down (1.2344 -> 1234), duration rounds up (0.0006s span -> 1ms rather
    # than truncating to 0).
    events = [MelodyNoteEvent(start_time=1.2344, end_time=1.2350, pitch=60, amplitude=0.5)]
    notes = build_melody_notes(events, lane_count=1, bpm=120.0)

    assert notes[0].time == 1234
    assert notes[0].duration == 1
