# Dual-Gate sweep: time per point and the time estimate

A point takes much longer than exposure + settling. The 201 x 151 sweep started on
2026-11-12 (1 s exposure, 0.05 s settling, NIRvana camera) ran at 2.4 s per point:
about 20 h instead of the 9 h that exposure and settling alone suggest. Its 0.1 s test
sweeps that afternoon had the same fixed overhead of about 1.25 s per point, so the
overhead does not depend on exposure or map size. It is the camera round trip beyond
the exposure, the Keysight commands and the program; the saved files of those sweeps
cannot split it further. The Dual-Gate tab now records where each point's time goes
and estimates from the measured pace. The code is in [`sweep_timing.py`](../sweep_timing.py).

## What is recorded

For every point the sweep records, in seconds of active time (pauses excluded):

| Field | Step |
|---|---|
| `ramp_v1_s`, `ramp_v2_s` | gate ramp to the point: waiting at the ramp rate, plus the Keysight's setpoint and compliance checks. The first point of each row includes the ramp back from the end of the previous row (unless serpentine). |
| `settle_s` | settling time |
| `readback_v1_s`, `readback_v2_s` | gate current and voltage readbacks |
| `field_s` | magnet field readback (0 with no magnet) |
| `spectrum_s` | camera request, including the exposure |
| `point_s` | the whole point, from the end of the previous one |
| `paused_s` | time paused during the point (not included above) |
| `finished_at` | Unix time at which the point was complete |

They are saved in the `timing` group of the sweep's HDF5 file, as arrays shaped
like the gate maps (NaN where not measured). The group's attributes hold a
`summary` sentence and the mean of each part (`mean_camera_s`,
`mean_gate_ramps_s`, `mean_gate_readbacks_s`, `mean_settling_s`,
`mean_field_readback_s`, `mean_other_s`, `mean_point_s`). The same summary is
in the "Sweep Complete" message and in the Device Manager's application log:

    <N> points in <duration>, <t> s per point on average: camera <t> s, gate ramps <t> s,
    gate readbacks <t> s, settling <t> s, field readback <t> s, other <t> s.
    The camera time includes <exposure> s of exposure.

"other" is the rest of the point time: pause checks, signals and Python.

Reading the timing of a saved sweep:

```python
import h5py, numpy as np
with h5py.File("path/to/gate_sweep.h5") as f:
    t = f["timing"]
    print(t.attrs["summary"])
    rows = np.nanmedian(t["ramp_v1_s"][()], axis=1)  # e.g. the ramp per row
```

## Expected End

The label names the day, the total duration and the pace, for example
`Expected End: tomorrow 14:05 · 20 h 12 min in total · 2.39 s per point`.
It models a point as settling + frames x exposure + |gate step| / ramp rate per
gate + an overhead per point:

1. Before the first points it starts from the overhead of the last measured
   sweep (`dual_gate.point_overhead_s` in `value_storage`), marked "pace of the
   last sweep". With none stored, it uses the old fixed guess (2.05 s per frame
   plus 0.13 s) and says "first guess".
2. From the third point on, the overhead is this sweep's own: the time so far
   minus the modelled exposure, settling and ramp waits, per point. The estimate
   therefore follows the real pace, without a limit on the correction.
3. The measured overhead is kept for the next sweep only after 20 points, and only
   when the camera is physical: a simulated camera has no exposure time.

Pauses are left out of the total; the end time moves with them.

Position DGS uses the same sweep loop but shows its point count across positions
and does not save these timings.

Settling now waits exactly the set time. It used to round up to the next 0.05 s
check, so a 0.02 s setting waited 0.05 s.
