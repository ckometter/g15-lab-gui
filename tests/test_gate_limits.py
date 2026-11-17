"""Per-gate voltage windows for Dual Gate sweeps, with fake instruments only."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6 import QtCore

from gate_safety import DEFAULT_GATE_LIMITS_V, GATE_MAX_ABS_VOLTAGE_V, check_gate_limits, validate_gate_limits
from measurements import Dual_gate_sweep_gui as dgs
from position_dgs import PositionSweepWorker

DIRECT = QtCore.Qt.ConnectionType.DirectConnection


class GateLimitCheckTests(unittest.TestCase):
    def test_window_includes_0_v_and_stays_within_the_smu_range(self):
        self.assertEqual(validate_gate_limits(*DEFAULT_GATE_LIMITS_V), (-1.0, 1.0))
        self.assertEqual(validate_gate_limits("-3", 4, 1), (-3.0, 4.0))
        self.assertEqual(validate_gate_limits(0, 0), (0.0, 0.0))
        self.assertEqual(validate_gate_limits(-GATE_MAX_ABS_VOLTAGE_V, GATE_MAX_ABS_VOLTAGE_V),
                         (-GATE_MAX_ABS_VOLTAGE_V, GATE_MAX_ABS_VOLTAGE_V))
        for lowest, highest, side in ((0.5, 4, "lowest"), (-GATE_MAX_ABS_VOLTAGE_V - 1, 4, "lowest"),
                                      (float("nan"), 4, "lowest"), (-3, -0.5, "highest"),
                                      (-3, GATE_MAX_ABS_VOLTAGE_V + 1, "highest"), (-3, float("inf"), "highest")):
            with self.subTest(lowest=lowest, highest=highest), \
                    self.assertRaisesRegex(ValueError, f"Channel 2: the {side} voltage must be between"):
                validate_gate_limits(lowest, highest, 2)

    def test_targets_outside_a_window_name_the_channel_the_side_and_the_voltage(self):
        limits = {1: (-3.0, 4.0), 2: (-8.0, 2.0)}
        check_gate_limits([-3.0, 0.0, 4.0], [-8.0, 2.0], limits)
        check_gate_limits([4.0 + 1e-12, -3.0 - 1e-12], [], limits)  # Rounding in the matrix.
        # The target farthest outside is named: 4.2 V is 0.2 V above, -3.1 V only 0.1 V below.
        with self.assertRaisesRegex(ValueError, r"^Channel 1 would reach \+4\.2 V, above its highest allowed \+4 V\.$"):
            check_gate_limits([1.0, -3.1, 4.2], [0.0], limits)
        with self.assertRaisesRegex(ValueError, r"^Channel 1 would reach -3\.5 V, below its lowest allowed -3 V\.$"):
            check_gate_limits([-3.5, 4.2], [0.0], limits)
        with self.assertRaisesRegex(ValueError, r"^Channel 2 would reach \+2\.5 V, above its highest allowed \+2 V\.$"):
            check_gate_limits([0.0], np.array([[2.5, -1.0]]), limits)

    def test_a_window_may_allow_more_on_one_side(self):
        check_gate_limits([-7.5, 1.9], [], {1: (-8.0, 2.0)})
        with self.assertRaisesRegex(ValueError, r"above its highest allowed \+2 V"):
            check_gate_limits([2.5], [], {1: (-8.0, 2.0)})

    def test_a_channel_without_a_window_is_not_checked_here(self):
        check_gate_limits([100.0], [100.0], None)
        check_gate_limits([100.0], [1.0], {2: (-4.0, 4.0)})

    def test_matrix_corners_outside_a_window_block_the_sweep(self):
        # VG1 = V1 + V2 and VG2 = V1 - V2, so V1 = (VG1 + VG2) / 2 and V2 = (VG1 - VG2) / 2:
        # over this rectangle both range from -1 V to +5 V.
        params = dict(a=1., b=1., c=1., d=-1., c1_start=0., c1_end=8., c1_pts=5,
                      c2_start=-2., c2_end=2., c2_pts=3)
        dgs.validate_dgs_parameters({**params, 'gate_limits': {1: (-1.0, 5.0), 2: (-1.0, 5.0)}})
        with self.assertRaisesRegex(ValueError, r"Channel 1 would reach -1 V, below its lowest allowed -0\.5 V"):
            dgs.validate_dgs_parameters({**params, 'gate_limits': {1: (-0.5, 5.0), 2: (-1.0, 5.0)}})
        with self.assertRaisesRegex(ValueError, r"Channel 2 would reach \+5 V, above its highest allowed \+4 V"):
            dgs.validate_dgs_parameters({**params, 'gate_limits': {1: (-1.0, 5.0), 2: (-1.0, 4.0)}})


class SweepWorkerLimitTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(a=1., b=0., c=0., d=1., c1_start=0., c1_end=.3, c1_pts=4,
                           c2_start=0., c2_end=.1, c2_pts=2, sweep_mode='Hold C2, sweep C1',
                           serpentine=False, exposure_time=.01, settling_time=0., num_pixels=3,
                           num_spectra=1, gate_limits={1: (-1.0, 0.25), 2: (-1.0, 1.0)})
        self.station = SimpleNamespace(smu=Mock(), camera=Mock(), monochromator=Mock(),
                                       piezo300=Mock(spec=['get_axis_status', 'get_voltage', 'set_voltage']))
        for axis in (1, 2):
            channel = getattr(self.station.smu, f'ch{axis}')
            channel.source_mode.return_value = 'VOLT'
            channel.volt.return_value = 0.
            channel.ramp_stop_requested = False
            channel.output.return_value = True
            channel.measure_curr.return_value = 0.
            channel.measure_volt.return_value = 0.
        offsets = {1: 9., 2: 32.}
        self.station.piezo300.get_axis_status.side_effect = lambda axis: {'offset': offsets[axis], 'mode': 'off'}
        self.station.piezo300.get_voltage.side_effect = offsets.__getitem__
        self.enterContext(patch.object(dgs, 'acquire_sweep_spectrum', return_value=(np.ones(3), np.ones((1, 3)))))
        self.enterContext(patch.object(dgs.os.path, 'exists', return_value=False))

    def test_sweep_beyond_a_limit_never_writes(self):
        worker = dgs.SweepWorker(self.station, self.params)
        errors, points = [], []
        worker.error_occurred.connect(errors.append, DIRECT)
        worker.point_measured.connect(lambda *point: points.append(point), DIRECT)
        worker._do_run()
        self.assertIn("Channel 1 would reach +0.3 V, above its highest allowed +0.25 V", " ".join(errors))
        self.assertEqual(points, [])
        self.station.smu.ch1.set_r.volt.assert_not_called()
        self.station.smu.ch2.set_r.volt.assert_not_called()
        self.station.camera.exposure_time.assert_not_called()

    def test_each_point_is_checked_again_before_its_ramp(self):
        # Even if the preflight were bypassed, the first point beyond the limit stops the sweep.
        worker = dgs.SweepWorker(self.station, self.params)
        errors, points = [], []
        worker.error_occurred.connect(errors.append, DIRECT)
        worker.point_measured.connect(lambda *point: points.append(point), DIRECT)
        with patch.object(dgs, 'validate_dgs_parameters'):
            worker._do_run()
        self.assertIn("Channel 1 would reach +0.3 V", " ".join(errors))
        self.assertEqual(len(points), 3)  # 0, 0.1 and 0.2 V; not 0.3 V
        targets = [call.args[0] for call in self.station.smu.ch1.set_r.volt.call_args_list]
        self.assertLessEqual(max(targets), 0.25)

    def test_negative_side_has_its_own_limit(self):
        self.params.update(c1_start=-0.3, c1_end=0.3, gate_limits={1: (-0.2, 1.0), 2: (-1.0, 1.0)})
        worker = dgs.SweepWorker(self.station, self.params)
        errors = []
        worker.error_occurred.connect(errors.append, DIRECT)
        worker._do_run()
        self.assertIn("Channel 1 would reach -0.3 V, below its lowest allowed -0.2 V", " ".join(errors))
        self.station.smu.ch1.set_r.volt.assert_not_called()

    def test_sweep_within_the_limits_runs_every_point(self):
        self.params['gate_limits'] = {1: (0.0, 0.3), 2: (0.0, 0.1)}
        worker = dgs.SweepWorker(self.station, self.params)
        errors, points = [], []
        worker.error_occurred.connect(errors.append, DIRECT)
        worker.point_measured.connect(lambda *point: points.append(point), DIRECT)
        worker._do_run()
        self.assertEqual(errors, [])
        self.assertEqual(len(points), 8)

    def test_position_scan_beyond_a_limit_never_moves(self):
        scan = dict(x=np.array([9.]), y=np.array([32.]), move_rate=.5, return_rate=.2, settle=0)
        store = Mock()
        worker = PositionSweepWorker(self.station, self.params, scan, store)
        worker.run()
        self.assertIn("Channel 1 would reach +0.3 V", worker.failure)
        self.station.smu.ch1.set_r.volt.assert_not_called()
        self.station.piezo300.set_voltage.assert_not_called()


if __name__ == '__main__':
    unittest.main()
