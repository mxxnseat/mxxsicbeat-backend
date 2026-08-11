import bisect

import numpy as np

from app.domains.maps.dsp.kick_onseter import KickOnseter
from app.domains.maps.dsp.melody_extractor import MelodyNoteEvent
from app.domains.maps.dtos.notes import Note, NoteType

# Salience = amplitude scaled by held duration, capped so a long sustain doesn't keep gaining
# weight past what already reads as "clearly the important note" - without the cap, raw duration
# would dominate the score and let a slow, uninteresting sustain outrank a short, loud, on-beat hit.
_SALIENCE_DURATION_CAP_S = 0.3

# A note within this many semitones of a lane-band boundary keeps the previous note's lane
# instead of re-deciding from scratch - basic-pitch's pitch estimate is noisy enough that notes
# hovering right at the boundary would otherwise flip lanes independent of any real melodic
# movement.
_LANE_HYSTERESIS_SEMITONES = 1

# Max one kept note per lane per this fraction of a beat, derived from bpm - caps density to
# something tappable and collapses basic-pitch's over-transcription (grace notes, vibrato
# artifacts, chord notes sharing a band) down to the single most salient note in each slot.
_SUPPRESSION_WINDOW_BEAT_FRACTION = 8

# Minimum time a held note's end must leave before the next note in the same lane starts, so a
# player can release and catch that next note rather than the hold running straight into it.
# Density suppression alone doesn't guarantee this - it only caps how many onsets survive per
# window, not how long a surviving note's hold eats into the next slot.
_MIN_LANE_GAP_S = 0.2


def build_drum_notes(onset_times: list[float], onseter: KickOnseter, lane_count: int) -> list[Note]:
    """Turn detected kick onsets into lane-assigned notes.

    Lane assignment is a simple round-robin over onset order - there's no pitch or
    frequency-band signal yet to place notes more meaningfully (only kick-onset detection is
    implemented so far).
    """
    return [
        Note(
            time=round(onset_time * 1000),
            lane=index % lane_count,
            energy=onseter.strength_at(onset_time),
            note_type=NoteType.DRUM,
        )
        for index, onset_time in enumerate(onset_times)
    ]


def _salience(event: MelodyNoteEvent) -> float:
    duration = event.end_time - event.start_time
    return event.amplitude * min(duration, _SALIENCE_DURATION_CAP_S) / _SALIENCE_DURATION_CAP_S


def _lane_boundaries(pitches: list[int], lane_count: int) -> list[float]:
    """`lane_count - 1` quantile cut points splitting `pitches` into `lane_count` equal-sized
    bands (lowest pitches in lane 0), so lane density stays balanced regardless of where the
    melody's register sits - a fixed pitch cutoff would badly mis-split a bass-heavy vs a
    soprano-heavy track."""
    if lane_count <= 1 or not pitches:
        return []
    quantiles = [band / lane_count for band in range(1, lane_count)]
    return list(np.quantile(pitches, quantiles))


def _assign_lanes(events: list[MelodyNoteEvent], lane_count: int) -> list[int]:
    """Pitch-band lane assignment: splits notes into `lane_count` bands by pitch quantile, then
    walks notes in time order applying a hysteresis margin around each boundary so a note near
    the edge inherits the previous note's lane instead of flipping on noisy pitch estimation
    alone. A chord - multiple simultaneous notes spanning more than one band - naturally lands
    in multiple lanes at the same timestamp this way, with no special-casing needed."""
    if lane_count <= 1:
        return [0] * len(events)

    order = sorted(range(len(events)), key=lambda i: events[i].start_time)
    boundaries = _lane_boundaries([events[i].pitch for i in order], lane_count)

    lanes = [0] * len(events)
    prev_lane: int | None = None
    for i in order:
        pitch = events[i].pitch
        band = bisect.bisect_left(boundaries, pitch)
        near_boundary = any(abs(pitch - boundary) <= _LANE_HYSTERESIS_SEMITONES for boundary in boundaries)
        lane = prev_lane if prev_lane is not None and near_boundary else band
        lanes[i] = lane
        prev_lane = lane
    return lanes


def _suppress_by_window(
    events: list[MelodyNoteEvent], lanes: list[int], salience: list[float], window_seconds: float
) -> list[int]:
    """Within each (lane, time window) bucket, keep only the most salient note's index - caps
    density per lane while letting different lanes independently keep simultaneous notes."""
    best_index_by_bucket: dict[tuple[int, int], int] = {}
    for i, event in enumerate(events):
        bucket = (lanes[i], int(event.start_time // window_seconds))
        current = best_index_by_bucket.get(bucket)
        if current is None or salience[i] > salience[current]:
            best_index_by_bucket[bucket] = i
    return sorted(best_index_by_bucket.values(), key=lambda i: events[i].start_time)


def _clamp_hold_gaps(events: list[MelodyNoteEvent], kept: list[int], lanes: list[int]) -> dict[int, float]:
    """Shortens a held note's end time when it runs past (or too close to) the start of the next
    note in the same lane, leaving `_MIN_LANE_GAP_S` free to release and catch that next note.
    Only ever pulls a note's end time earlier, floored at its own start - onset times and lane
    assignment (the actual clustering) are untouched, so this can't undo what `_assign_lanes` or
    `_suppress_by_window` decided, only trim how long a hold visually/mechanically lasts."""
    end_times = {i: events[i].end_time for i in kept}

    by_lane: dict[int, list[int]] = {}
    for i in kept:
        by_lane.setdefault(lanes[i], []).append(i)

    for lane_indices in by_lane.values():
        lane_indices.sort(key=lambda i: events[i].start_time)
        for current, following in zip(lane_indices, lane_indices[1:], strict=False):
            max_end = events[following].start_time - _MIN_LANE_GAP_S
            end_times[current] = max(events[current].start_time, min(end_times[current], max_end))

    return end_times


def build_melody_notes(note_events: list[MelodyNoteEvent], lane_count: int, bpm: float) -> list[Note]:
    """Turn basic-pitch's transcribed note events into lane-assigned notes.

    Lanes are pitch bands rather than round-robin order (see `_assign_lanes`), and density is
    capped per lane per beat-aligned window (see `_suppress_by_window`) using each note's
    salience - amplitude scaled by held duration - so the busiest, noisiest transcription still
    collapses to a tappable rhythm without losing legitimate chords (simultaneous notes in
    different bands survive independently). A held note that would still run into the next
    note in the same lane after all that has its end trimmed by `_clamp_hold_gaps`.
    """
    if not note_events:
        return []

    lanes = _assign_lanes(note_events, lane_count)
    salience = [_salience(event) for event in note_events]
    window_seconds = (60.0 / bpm) / _SUPPRESSION_WINDOW_BEAT_FRACTION
    kept = _suppress_by_window(note_events, lanes, salience, window_seconds)
    end_times = _clamp_hold_gaps(note_events, kept, lanes)

    return [
        Note(
            time=round(note_events[i].start_time * 1000),
            duration=round((end_times[i] - note_events[i].start_time) * 1000),
            lane=lanes[i],
            energy=note_events[i].amplitude,
            note_type=NoteType.MELODY,
        )
        for i in kept
    ]
