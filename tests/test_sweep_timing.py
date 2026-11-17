"""Per-point timing of dual-gate sweeps and the time estimate built on it. No hardware."""
import json
import math
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6 import QtCore

import sweep_timing as st
from measurements import Dual_gate_sweep_gui as dgs

DIRECT = QtCore.Qt.ConnectionType.DirectConnection


class EstimateTests(unittest.TestCase):
    def test_ramp_waits_include_the_first_ramp_and_every_row_return(self):
        # Two rows of three points along V1 at 1 V/s; V2 steps by 0.5 V at 0.5 V/s.
        points = [(0, 0), (1, 0), (2, 0), (0, 0.5), (1, 0.5), (2, 0.5)]
        waits = st.ramp_waits(points, start=(-1, 0), rates=(1.0, 0.5))
        np.testing.assert_allclose(waits, [1, 1, 1, 2 + 1, 1, 1])

    def test_a_gate_without_a_ramp_rate_or_start_adds_no_wait(self):
        np.testing.assert_allclose(st.ramp_waits([(1, 5)], (0, 0), (1.0, 0.0)), [1.0])
        np.testing.assert_allclose(st.ramp_waits([(1, 5)], (0, math.nan), (1.0, 1.0)), [1.0])
        self.assertEqual(st.ramp_waits([], (0, 0), (1, 1)).size, 0)

    def test_estimate_takes_the_overhead_from_the_measured_pace(self):
        # Like the 201 x 151 sweep: 1 s exposure, 0.05 s settling and 1.25 s of
        # overhead per point, where the old fixed guess was 2.18 s.
        points = [(v, 0.0) for v in np.linspace(0, 1, 100)]
        rates = (1.0, 1.0)
        actual = 1.0 + 0.05 + 1.25 + st.ramp_waits(points, (0, 0), rates)
        estimate = st.SweepEstimate(points, (0, 0), rates, exposure=1.0, frames=1, settling=0.05,
                                    overhead=st.default_point_overhead(1))
        self.assertEqual(estimate.basis, 'default')
        self.assertGreater(estimate.total(), actual.sum() + 90)
        for done in (1, 2):  # Too few points: the first guess stays.
            estimate.update(done, actual[:done].sum())
            self.assertEqual(estimate.basis, 'default')
        left = estimate.update(10, actual[:10].sum())
        self.assertEqual(estimate.basis, 'measured')
        self.assertAlmostEqual(estimate.overhead, 1.25)
        self.assertAlmostEqual(left, actual[10:].sum())
        self.assertAlmostEqual(estimate.total(), actual.sum())
        self.assertEqual(estimate.update(100, actual.sum()), 0.0)

    def test_frames_multiply_the_exposure_and_the_old_readout_guess(self):
        estimate = st.SweepEstimate([(0, 0)] * 4, (0, 0), (1, 1), exposure=0.5, frames=3,
                                    settling=0.1, overhead=1.0)
        self.assertAlmostEqual(estimate.total(), 4 * (0.1 + 1.5 + 1.0))
        self.assertAlmostEqual(st.default_point_overhead(3), 3 * 2.05 + 0.13)

    def test_expected_end_names_the_day_the_duration_and_the_pace(self):
        start = datetime(2026, 11, 12, 17, 52)
        hours = 20.2 * 3600
        self.assertEqual(st.expected_end_text(hours, hours, 30351, now=start.timestamp()),
                         'Expected End: tomorrow 14:04 · 20 h 12 min in total · 2.40 s per point')
        self.assertEqual(st.expected_end_text(600, 3600, 1800, now=start.timestamp(), basis='stored'),
                         'Expected End: about today 18:02 · 1 h 0 min in total · 2.00 s per point'
                         ' (pace of the last sweep)')
        later = start + timedelta(days=4)
        self.assertEqual(st.expected_end_text(4 * 86400, 4 * 86400, 1, now=start.timestamp(), basis='default'),
                         f"Expected End: about {later:%a %d %b} 17:52 · 4 d 0 h in total · 345600.00 s "
                         f"per point (first guess, until a few points are measured)")

    def test_durations_read_naturally(self):
        for seconds, text in ((-3, '0 s'), (45, '45 s'), (59.4, '59 s'), (59.6, '1 min'), (725, '12 min'),
                              (72720, '20 h 12 min'), (2 * 86400 + 3 * 3600, '2 d 3 h')):
            with self.subTest(seconds=seconds):
                self.assertEqual(st.format_duration(seconds), text)

    def test_summary_splits_the_mean_point_time_into_its_parts(self):
        timing = {name: np.array([np.nan, np.nan, np.nan]) for name in st.TIMING_FIELDS}
        for name, value in (('spectrum_s', 1.9), ('ramp_v1_s', 0.1), ('ramp_v2_s', 0.05),
                            ('readback_v1_s', 0.15), ('readback_v2_s', 0.1), ('settle_s', 0.05),
                            ('field_s', 0.0), ('point_s', 2.4), ('paused_s', 0.0)):
            timing[name][:2] = value
        summary = st.timing_summary(timing, exposure=1.0, frames=1)
        self.assertEqual(summary['points'], 2)
        self.assertAlmostEqual(summary['point_s'], 2.4)
        self.assertAlmostEqual(summary['parts']['other'], 0.05)
        self.assertAlmostEqual(sum(summary['parts'].values()), 2.4)
        self.assertEqual(st.summary_text(summary),
                         '2 points in 5 s, 2.40 s per point on average: camera 1.90 s, gate ramps 0.15 s, '
                         'gate readbacks 0.25 s, settling 0.05 s, field readback 0.00 s, other 0.05 s. '
                         'The camera time includes 1 s of exposure.')
        self.assertIsNone(st.timing_summary({name: np.full(3, np.nan) for name in st.TIMING_FIELDS},
                                            exposure=1.0, frames=1))

    def test_only_a_plausible_stored_overhead_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'value_storage'
            self.assertIsNone(st.stored_point_overhead(path))
            for value, expected in ((1.25, 1.25), (0, 0.0), (-1, None), ('fast', None), (None, None),
                                    (float('nan'), None), (1e6, None)):
                with self.subTest(value=value):
                    path.write_text(json.dumps({'dual_gate': {st.STORAGE_KEY: value}}))
                    self.assertEqual(st.stored_point_overhead(path), expected)
            path.write_text('not json')
            self.assertIsNone(st.stored_point_overhead(path))


def sleeper(seconds, result=None):
    def call(*args, **kwargs):
        time.sleep(seconds)
        return result
    return call


class SweepWorkerTimingTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(a=1., b=0., c=0., d=1., c1_start=0., c1_end=.1, c1_pts=2,
                           c2_start=0., c2_end=.1, c2_pts=2, sweep_mode='Hold C2, sweep C1',
                           serpentine=False, exposure_time=.01, settling_time=.02,
                           num_pixels=3, num_spectra=1)
        self.station = SimpleNamespace(smu=Mock(), camera=Mock(), monochromator=Mock(), magnet=Mock())
        for axis, ramp in ((1, .02), (2, .01)):
            channel = getattr(self.station.smu, f'ch{axis}')
            channel.source_mode.return_value = 'VOLT'
            channel.volt.return_value = 0.
            channel.ramp_stop_requested = False
            channel.output.return_value = True
            channel.set_r.volt.side_effect = sleeper(ramp)
            channel.measure_curr.side_effect = sleeper(.005, 1e-10)
            channel.measure_volt.side_effect = sleeper(.005, 0.)
        self.station.magnet.output_field.side_effect = sleeper(.01, 0.)
        self.acquisition = self.enterContext(patch.object(
            dgs, 'acquire_sweep_spectrum', side_effect=sleeper(.03, (np.ones(3), np.ones((1, 3))))))
        # Persisted acquisition settings must not change the test.
        self.enterContext(patch.object(dgs.os.path, 'exists', return_value=False))

    def run_worker(self):
        worker = dgs.SweepWorker(self.station, self.params)
        worker.timings, worker.points, worker.errors = [], [], []
        worker.point_timing.connect(lambda i, j, timing: worker.timings.append((i, j, timing)), DIRECT)
        worker.point_measured.connect(lambda *point: worker.points.append(point[:2]), DIRECT)
        worker.error_occurred.connect(worker.errors.append, DIRECT)
        started = time.time()
        worker._do_run()
        self.assertEqual(worker.errors, [])
        return worker, started

    def test_every_point_records_how_long_each_step_took(self):
        worker, started = self.run_worker()
        self.assertEqual([(i, j) for i, j, _ in worker.timings], worker.points)
        self.assertEqual(len(worker.timings), 4)
        previous_end = started
        for _, _, timing in worker.timings:
            self.assertEqual(set(timing), set(st.TIMING_FIELDS))
            for name, minimum in (('ramp_v1_s', .02), ('ramp_v2_s', .01), ('settle_s', .02),
                                  ('readback_v1_s', .01), ('readback_v2_s', .01), ('field_s', .01),
                                  ('spectrum_s', .03)):
                self.assertGreaterEqual(timing[name], minimum * 0.9, name)
            phases = sum(timing[name] for name in st.PHASE_FIELDS)
            self.assertGreaterEqual(timing['point_s'], phases)
            self.assertLess(timing['point_s'], phases + .05)
            self.assertEqual(timing['paused_s'], 0.0)
            self.assertGreaterEqual(timing['finished_at'], previous_end)
            previous_end = timing['finished_at']
        # Settling is the setting, not rounded up to the 0.05 s check interval.
        self.assertLess(min(timing['settle_s'] for _, _, timing in worker.timings), .045)

    def test_a_pause_is_recorded_apart_and_left_out_of_the_steps(self):
        first = {'done': False}

        def acquire_then_pause(*args, **kwargs):
            if not first['done']:
                first['done'] = True
                worker.pause()
                threading.Timer(.3, worker.resume).start()
            return np.ones(3), np.ones((1, 3))

        self.acquisition.side_effect = acquire_then_pause
        worker = dgs.SweepWorker(self.station, self.params)
        timings, errors = [], []
        worker.point_timing.connect(lambda i, j, timing: timings.append(timing), DIRECT)
        worker.error_occurred.connect(errors.append, DIRECT)
        worker._do_run()
        self.assertEqual(errors, [])
        self.assertEqual(len(timings), 4)
        self.assertGreaterEqual(timings[0]['paused_s'], .25)
        self.assertLess(timings[0]['point_s'], .25)
        self.assertLess(max(timings[0][name] for name in st.PHASE_FIELDS), .25)
        self.assertTrue(all(timing['paused_s'] == 0.0 for timing in timings[1:]))
        self.assertAlmostEqual(worker.paused_seconds, timings[0]['paused_s'])


if __name__ == '__main__':
    unittest.main()
