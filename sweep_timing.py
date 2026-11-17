"""Per-point timing of dual-gate sweeps, and the time estimate built on it.

Every sweep point records how long each of its steps took (gate ramps, settling,
gate readbacks, field readback, spectrum), so a saved file shows where the time
went. The estimate models one point as

    settling + frames x exposure + ramp waits + overhead,

where the ramp waits are each gate's voltage change divided by its ramp rate
(including the ramp back at the start of every row) and the overhead is all the
rest: the camera round trip beyond the exposure, the Keysight commands and the
program itself. While a sweep runs, the overhead is taken from its measured pace.
The last measured value is kept in value_storage for the next sweep's first estimate.
"""
import json
import math
import time
from datetime import datetime, timedelta

import numpy as np

# Recorded for every point: seconds of active time (pauses excluded), except
# paused_s (the pause time within the point) and finished_at (Unix time at the end).
TIMING_FIELDS = ('ramp_v1_s', 'ramp_v2_s', 'settle_s', 'readback_v1_s', 'readback_v2_s',
                 'field_s', 'spectrum_s', 'point_s', 'paused_s', 'finished_at')
PHASE_FIELDS = TIMING_FIELDS[:7]
# How a summary groups the phases. "other" is the rest of the point time.
SUMMARY_PARTS = (('camera', ('spectrum_s',)),
                 ('gate ramps', ('ramp_v1_s', 'ramp_v2_s')),
                 ('gate readbacks', ('readback_v1_s', 'readback_v2_s')),
                 ('settling', ('settle_s',)),
                 ('field readback', ('field_s',)))
STORAGE_KEY = 'point_overhead_s'
# The live estimate uses the sweep's own pace from this many points on.
MIN_POINTS_FOR_PACE = 3
# A measured overhead is kept for the next sweep only after this many points.
MIN_POINTS_TO_STORE = 20
BASIS_NOTES = {
    'measured': '',
    'stored': ' (pace of the last sweep)',
    'default': ' (first guess, until a few points are measured)',
}


def default_point_overhead(frames=1):
    """The earlier fixed guess, used until a sweep has been measured: 2.05 s of CCD
    readout per frame, 0.05 s of communication and 0.04 s of commands per gate."""
    return 2.05 * max(1, int(frames)) + 0.05 + 2 * 0.04


def ramp_waits(v_points, start, rates):
    """Seconds each point waits for its gates: |voltage change| / ramp rate, per gate.

    The first point ramps from ``start``. The first point of every row includes the
    ramp back from the end of the previous row. A gate without a positive rate adds
    no wait.
    """
    points = np.asarray(v_points, dtype=float).reshape(-1, 2)
    if not len(points):
        return np.zeros(0)
    previous = np.vstack((np.asarray(start, dtype=float).reshape(1, 2), points[:-1]))
    rates = np.asarray(rates, dtype=float).reshape(2)
    usable = np.isfinite(rates) & (rates > 0)
    waits = np.zeros_like(points)
    waits[:, usable] = np.abs(points - previous)[:, usable] / rates[usable]
    return np.where(np.isfinite(waits), waits, 0.0).sum(axis=1)


class SweepEstimate:
    """Expected duration of a sweep, refined from its own measured pace."""

    def __init__(self, v_points, start, rates, *, exposure, frames, settling, overhead, basis='default'):
        self.exposure, self.frames = float(exposure), max(1, int(frames))
        base = float(settling) + self.frames * self.exposure + ramp_waits(v_points, start, rates)
        self.points = len(base)
        # cumulative[k]: the modelled seconds of the first k points, without overhead.
        self.cumulative = np.concatenate(([0.0], np.cumsum(base)))
        self.overhead = float(overhead)
        self.basis = basis

    def total(self):
        return float(self.cumulative[-1] + self.points * self.overhead)

    def update(self, done, active_elapsed):
        """Measure the overhead from the first ``done`` points; return the seconds left."""
        done = min(max(int(done), 0), self.points)
        if done >= MIN_POINTS_FOR_PACE and active_elapsed > 0:
            self.overhead = (float(active_elapsed) - self.cumulative[done]) / done
            self.basis = 'measured'
        left = self.cumulative[-1] - self.cumulative[done] + (self.points - done) * self.overhead
        return max(0.0, float(left))


def format_duration(seconds):
    """45 s, 12 min, 20 h 12 min or 2 d 3 h."""
    seconds = max(0.0, float(seconds))
    if seconds < 59.5:
        return f'{seconds:.0f} s'
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f'{minutes} min'
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f'{hours} h {minutes} min'
    days, hours = divmod(hours, 24)
    return f'{days} d {hours} h'


def clock_text(when, now):
    """'today 14:05', 'tomorrow 14:05' or 'Sat 14 Nov 14:05'."""
    end, today = datetime.fromtimestamp(when), datetime.fromtimestamp(now).date()
    day = {today: 'today', today + timedelta(days=1): 'tomorrow'}.get(end.date())
    return f"{day or end.strftime('%a %d %b')} {end:%H:%M}"


def expected_end_text(seconds_left, total_seconds, points, *, basis='measured', now=None):
    """The Expected End label: the end with its day, the total duration and the pace."""
    now = time.time() if now is None else now
    pace = total_seconds / points if points else 0.0
    about = '' if basis == 'measured' else 'about '
    return (f'Expected End: {about}{clock_text(now + seconds_left, now)} · '
            f'{format_duration(total_seconds)} in total · {pace:.2f} s per point'
            f'{BASIS_NOTES.get(basis, "")}')


def _mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def timing_summary(timing, *, exposure, frames):
    """Mean seconds per point of each part, over the measured points; None if none."""
    point = np.asarray(timing.get('point_s', ()), dtype=float)
    measured = np.isfinite(point)
    if not measured.any():
        return None
    parts = {label: sum(_mean(np.asarray(timing[name], dtype=float)[measured]) for name in names)
             for label, names in SUMMARY_PARTS}
    mean_point = float(point[measured].mean())
    parts['other'] = mean_point - sum(parts.values())
    return {'points': int(measured.sum()), 'total_s': float(point[measured].sum()),
            'point_s': mean_point, 'exposure_s': max(1, int(frames)) * float(exposure), 'parts': parts}


def summary_text(summary):
    parts = ', '.join(f'{label} {seconds:.2f} s' for label, seconds in summary['parts'].items())
    return (f"{summary['points']} points in {format_duration(summary['total_s'])}, "
            f"{summary['point_s']:.2f} s per point on average: {parts}. "
            f"The camera time includes {summary['exposure_s']:g} s of exposure.")


def stored_point_overhead(storage_path):
    """The per-point overhead measured in the last sweep (value_storage), or None."""
    try:
        with open(storage_path, 'r') as f:
            value = float(json.load(f).get('dual_gate', {}).get(STORAGE_KEY))
    except Exception:
        return None
    return value if math.isfinite(value) and 0 <= value < 3600 else None
