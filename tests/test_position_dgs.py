"""Safety and filename checks, without physical device access."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from PyQt6 import QtCore

from measurement_filenames import dgs_filename
from measurements.Dual_gate_sweep_gui import SweepWorker
from position_dgs import (PositionSweepWorker, scan_axis, ramp_scanner,
                          scanner_state, read_scanner_offsets)


class PositionDGSTests(unittest.TestCase):
    def test_requested_filename_and_fixed_gate_omission(self):
        self.assertEqual(dgs_filename(exposure=.2, acquisitions=1,
                         axes=[('Vg', np.linspace(-10, 10, 101)), ('VE', [3])]),
                         'DGS_RC_0T_0p2s_1a_1090nm_Vg_m10-10V_101x1_.h5')
        name = dgs_filename(exposure=.5, acquisitions=4, field=-2, wavelength=1200,
                            axes=[('Vμ', [-1, 1]), ('VE', [-2, 0, 2])],
                            positions=([9, 9.1], [32]))
        self.assertEqual(name, 'PSDGS_RC_m2T_0p5s_4a_1200nm_Vmu_m1-1V_VE_m2-2V_2x3_X9-9p1V_Y32V_2x1_.h5')

    def test_grid_has_exact_steps_and_rejects_out_of_bounds_before_movement(self):
        np.testing.assert_allclose(scan_axis(59.8, .2, .1), [59.8, 59.9, 60])
        for args in [(59.9, .2, .1), (-.1, .2, .1), (0, 1, .3), (0, 1, 0), (float('nan'), 0, 1)]:
            with self.assertRaises(ValueError): scan_axis(*args)

    def scanner(self, start):
        scanner = Mock(spec=['get_axis_status', 'get_voltage', 'set_voltage'])
        values = {1: start, 2: 32.0}
        scanner.get_axis_status.side_effect = lambda axis: {'offset': values[axis], 'mode': 'off'}
        scanner.get_voltage.side_effect = lambda axis: values[axis]
        scanner.set_voltage.side_effect = lambda axis, value: values.__setitem__(axis, value)
        return scanner

    def test_slow_return_reads_start_and_bounds_each_step_and_rate(self):
        scanner = self.scanner(9.1)
        waits = []
        self.assertTrue(ramp_scanner(scanner, 1, 9.0, .2, lambda: True, lambda seconds: waits.append(seconds) or True))
        targets = [c.args[1] for c in scanner.set_voltage.call_args_list]
        self.assertTrue(all(9 <= v <= 9.1 for v in targets))
        self.assertTrue(np.all(np.diff([9.1] + targets) < 0))
        self.assertAlmostEqual(sum(waits), .5)
        self.assertAlmostEqual(targets[-1], 9)

    def test_cancel_and_bad_modes_never_issue_movement(self):
        scanner = self.scanner(9)
        self.assertFalse(ramp_scanner(scanner, 1, 9.1, .2, lambda: False))
        scanner.set_voltage.assert_not_called()
        with self.assertRaises(ValueError): ramp_scanner(scanner, 1, 60.1, .2, lambda: True)
        scanner.get_axis_status.return_value = {'offset': 9, 'mode': 'gnd'}
        scanner.get_axis_status.side_effect = None
        with self.assertRaises(RuntimeError): scanner_state(scanner)
        scanner.set_voltage.assert_not_called()

    def test_readback_mismatch_stops_ramp(self):
        scanner = self.scanner(9)
        scanner.get_voltage.side_effect = lambda axis: 8
        with self.assertRaises(RuntimeError):
            ramp_scanner(scanner, 1, 9.1, .2, lambda: True, lambda seconds: True)
        self.assertEqual(scanner.set_voltage.call_count, 1)

    def test_offset_read_does_not_require_mode_or_movement_readiness(self):
        scanner = self.scanner(61)
        scanner.get_axis_status.side_effect = RuntimeError('Mode query unavailable')
        self.assertEqual(read_scanner_offsets(scanner), [61, 32])
        scanner.get_axis_status.assert_not_called()
        scanner.set_voltage.assert_not_called()
        with self.assertRaises(ValueError): scan_axis(61, 0, .1)

    def test_offset_read_rejects_invalid_readbacks_without_zero_fallback(self):
        scanner = self.scanner(9)
        scanner.get_voltage.side_effect = [9, float('nan')]
        with self.assertRaises(RuntimeError): read_scanner_offsets(scanner)
        scanner.set_voltage.assert_not_called()

    def test_each_pixel_repeats_standalone_gate_sequence_before_any_scanner_move(self):
        # Unequal gate dimensions and nonzero scanner offsets expose swapped axes.
        for mode in ('Hold C1, sweep C2', 'Hold C2, sweep C1'):
            for serpentine in (False, True):
                for matrix in (np.eye(2), np.array([[1., 1.], [1., -1.]])):
                    with self.subTest(mode=mode, serpentine=serpentine, matrix=matrix):
                        scanner = self.scanner(9)
                        smu = Mock()
                        voltages, events = [0., 0.], []
                        for index, channel in enumerate((smu.ch1, smu.ch2)):
                            def set_gate(value, index=index):
                                voltages[index] = value
                                events.append(('gate', index, value))
                            channel.set_r.volt.side_effect = set_gate
                            channel.measure_volt.side_effect = lambda index=index: voltages[index]
                            channel.measure_curr.return_value = 0.
                            channel.output.return_value = True
                            channel.source_mode.return_value = "VOLT"
                            channel.volt.return_value = 0.
                        station = SimpleNamespace(smu=smu, piezo300=scanner,
                                                  camera=Mock(), monochromator=Mock())
                        params = dict(zip(('a', 'b', 'c', 'd'), matrix.ravel()))
                        params.update(c1_start=-.02, c1_end=.02, c1_pts=3,
                                      c2_start=.03, c2_end=.04, c2_pts=2,
                                      sweep_mode=mode, serpentine=serpentine,
                                      exposure_time=.001, settling_time=0,
                                      num_pixels=3, num_spectra=1)
                        def acquire(*args, **kwargs):
                            events.append(('spectrum', *voltages))
                            return np.ones(3), np.ones((1, 3))
                        direct = QtCore.Qt.ConnectionType.DirectConnection
                        baseline, points = SweepWorker(station, params), []
                        baseline.point_measured.connect(lambda *point: points.append(point), direct)
                        errors = []
                        baseline.error_occurred.connect(errors.append, direct)
                        with patch('measurements.Dual_gate_sweep_gui.acquire_sweep_spectrum', side_effect=acquire):
                            baseline.run()
                            self.assertEqual(errors, [])
                            baseline_events = events.copy()
                            expected_indices = []
                            outer_count, inner_count = (3, 2) if mode == 'Hold C1, sweep C2' else (2, 3)
                            for outer in range(outer_count):
                                inner_indices = range(inner_count)
                                if serpentine and outer % 2: inner_indices = reversed(inner_indices)
                                expected_indices.extend((outer, inner) if mode == 'Hold C1, sweep C2'
                                                        else (inner, outer) for inner in inner_indices)
                            self.assertEqual([(p[0], p[1]) for p in points], expected_indices)
                            for point in points:
                                np.testing.assert_allclose(point[4:6], np.linalg.solve(matrix, point[2:4]))

                            # Require the entire baseline trace between every pair of moves.
                            scan = dict(x=np.array([9., 9.01, 9.02]), y=np.array([32., 32.01]),
                                        move_rate=.5, return_rate=.2, settle=0)
                            store = Mock()
                            store.position_finished.return_value = True
                            worker = PositionSweepWorker(station, params, scan, store)
                            events.clear()
                            started, finished = [], []
                            worker.position_started.connect(lambda iy, ix: (started.append((iy, ix)), events.clear()), direct)
                            worker.position_finished.connect(lambda iy, ix: finished.append((iy, ix)), direct)
                            setter = scanner.set_voltage.side_effect
                            def move(axis, target):
                                self.assertEqual(events, baseline_events)
                                self.assertEqual(started, finished)
                                if axis == 2: self.assertAlmostEqual(scanner.get_voltage(1), 9.)
                                setter(axis, target)
                            def ramp(*args):
                                return ramp_scanner(*args, wait=lambda seconds: True)
                            with patch.object(scanner, 'set_voltage', side_effect=move), \
                                 patch('position_dgs.ramp_scanner', side_effect=ramp):
                                worker.run()
                        self.assertEqual(worker.failure, '')
                        self.assertTrue(worker.completed)
                        self.assertEqual(started, [(iy, ix) for iy in range(2) for ix in range(3)])
                        self.assertEqual(started, finished)
                        self.assertEqual(store.write_point.call_count, 36)
                        for n, call in enumerate(store.write_point.call_args_list):
                            iy, ix, point = call.args
                            self.assertEqual((iy, ix), started[n // 6])
                            self.assertEqual(point[:6], points[n % 6][:6])
                        self.assertAlmostEqual(scanner.get_voltage(1), 9.)
                        self.assertAlmostEqual(scanner.get_voltage(2), 32.01)

    def test_incomplete_gate_sweep_stops_before_next_scanner_move(self):
        scanner = self.scanner(9)
        station = SimpleNamespace(smu=Mock(), piezo300=scanner)
        station.smu.ch1.source_mode.return_value = "VOLT"
        station.smu.ch2.source_mode.return_value = "VOLT"
        station.smu.ch1.volt.return_value = 0.
        station.smu.ch2.volt.return_value = 0.
        store = Mock()
        store.position_finished.return_value = False
        scan = dict(x=np.array([9., 9.01]), y=np.array([32., 32.01]),
                    move_rate=.5, return_rate=.2, settle=0)
        worker = PositionSweepWorker(station, dict(a=1, b=0, c=0, d=1, c1_start=0, c1_end=.1, c1_pts=3, c2_start=0, c2_end=.1, c2_pts=2), scan, store)
        # Simulate a gate run returning early without signalling its error.
        with patch.object(SweepWorker, '_do_run'):
            worker.run()
        self.assertIn('Incomplete DGS', worker.failure)
        self.assertFalse(worker.completed)
        scanner.set_voltage.assert_not_called()
        store.position_started.assert_called_once_with(0, 0, [9., 32.])
        self.assertEqual(store.set_status.call_args.args[0], 'error')


if __name__ == '__main__': unittest.main()
