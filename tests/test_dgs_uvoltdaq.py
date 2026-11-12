"""Dual-gate sweeps on uVoltDAQ outputs: simulated DAC, fake Keysight, no hardware."""
import json
import math
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import h5py
import numpy as np
from PyQt6 import QtCore, QtWidgets

import test_dashboard_workflow as dashboard_checks
from dgs_gate_outputs import (DEFAULT_GATE_OUTPUTS, DacGateChannel, normalize_gate_outputs,
                              validate_gate_output_targets)
from drivers.qpg_uvoltdaq import DAC_RANGE_V, QPGuVoltDAQ
from gate_safety import GateRampCancelled
from measurements import Dual_gate_sweep_gui as dgs
from position_dgs import PositionSweepWorker

DIRECT = QtCore.Qt.ConnectionType.DirectConnection


class GateOutputTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("pyvisa.ResourceManager", side_effect=AssertionError("Hardware forbidden")))
        self.dac = QPGuVoltDAQ("test_gate_output_dac", address="simulated")
        self.addCleanup(self.dac.close)

    def test_outputs_default_to_the_keysight_and_reject_invalid_choices(self):
        self.assertEqual(normalize_gate_outputs(None), DEFAULT_GATE_OUTPUTS)
        self.assertEqual(normalize_gate_outputs({"1": "dac:0", "2": "smu:2"}), {1: "dac:0", 2: "smu:2"})
        for outputs, message in (({1: "smu:2"}, "unknown"), ({1: "dac:4"}, "unknown"),
                                 ({1: "dac:1", 2: "dac:1"}, "cannot both")):
            with self.subTest(outputs=outputs), self.assertRaisesRegex(ValueError, message):
                normalize_gate_outputs(outputs)

    def test_only_dac_gates_are_limited_to_the_dac_range(self):
        validate_gate_output_targets([50.0], [-50.0], DEFAULT_GATE_OUTPUTS)
        validate_gate_output_targets([-DAC_RANGE_V, DAC_RANGE_V], [50.0], {1: "dac:0", 2: "smu:2"})
        with self.assertRaisesRegex(ValueError, "V2 is driven by uVoltDAQ DAC3.*±10 V"):
            validate_gate_output_targets([0.0], [0.0, DAC_RANGE_V + 0.001], {1: "smu:1", 2: "dac:3"})
        with self.assertRaisesRegex(ValueError, "V1"):
            validate_gate_output_targets([math.nan], [0.0], {1: "dac:0", 2: "smu:2"})

    def test_dac_channel_offers_the_keysight_calls_the_sweep_uses(self):
        gate = DacGateChannel(self.dac, 2, axis=1, ramp_rate=0.5)
        progress = []
        gate.set_r.volt(0.12, callback=progress.append)
        self.assertAlmostEqual(self.dac.dac2.voltage(), 0.12)
        self.assertAlmostEqual(gate.volt(), 0.12)
        self.assertAlmostEqual(gate.measure_volt(), 0.12)  # Setpoint readback
        self.assertEqual(progress[-1], 0.12)
        self.assertTrue(all(abs(b - a) <= 0.05 + 1e-12 for a, b in zip([0.0] + progress, progress)))
        self.assertTrue(math.isnan(gate.measure_curr()))
        self.assertTrue(gate.output())
        self.assertEqual(gate.source_mode(), "VOLT")
        self.assertEqual(self.dac.dac2.ramp_rate(), 0.5)
        self.assertEqual(gate.channel_num, 1)
        with self.assertRaisesRegex(RuntimeError, "no output switch"):
            gate.output(False)
        with self.assertRaisesRegex(RuntimeError, "voltage"):
            gate.source_mode("CURR")
        gate.parent.return_to_local()

    def test_stop_flag_is_the_dac_channel_flag_and_cancels_a_ramp(self):
        gate = DacGateChannel(self.dac, 0, axis=2)
        gate.ramp_stop_requested = True
        self.assertTrue(self.dac.dac0.ramp_stop_requested)
        with self.assertRaises(GateRampCancelled):
            gate.set_r.volt(1.0)
        self.assertEqual(self.dac.dac0.voltage(), 0.0)
        gate.ramp_stop_requested = False
        self.assertFalse(self.dac.dac0.ramp_stop_requested)

    def test_out_of_range_dac_target_is_rejected_without_a_write(self):
        gate = DacGateChannel(self.dac, 1, axis=1)
        with self.assertRaisesRegex(ValueError, "±10 V"):
            gate.set_r.volt(DAC_RANGE_V + 1)
        self.assertEqual(self.dac.dac1.voltage(), 0.0)


class DacSweepWorkerTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("pyvisa.ResourceManager", side_effect=AssertionError("Hardware forbidden")))
        self.params = dict(a=1., b=0., c=0., d=1.,
                           c1_start=.1, c1_end=.2, c1_pts=2,
                           c2_start=-.3, c2_end=-.3, c2_pts=1,
                           sweep_mode='Hold C2, sweep C1', serpentine=False,
                           exposure_time=.01, settling_time=0., num_pixels=3, num_spectra=1,
                           gate_outputs={1: 'dac:0', 2: 'dac:3'}, gate_ramp_rates={1: .2, 2: .3})
        self.dac = QPGuVoltDAQ("test_dgs_worker_dac", address="simulated")
        self.addCleanup(self.dac.close)
        self.station = SimpleNamespace(smu=Mock(), camera=Mock(), monochromator=Mock(), dac=self.dac,
                                       piezo300=Mock(spec=['get_axis_status', 'get_voltage', 'set_voltage']))
        for axis in (1, 2):
            channel = getattr(self.station.smu, f'ch{axis}')
            channel.source_mode.return_value = 'VOLT'
            channel.volt.return_value = 0.
            channel.ramp_stop_requested = False
            channel.measure_curr.return_value = 2e-9
            channel.measure_volt.return_value = 0.
            channel.output.return_value = True
        self.acquisition = self.enterContext(patch.object(dgs, 'acquire_sweep_spectrum',
                                                          return_value=(np.ones(3), np.ones((1, 3)))))
        self.enterContext(patch.object(dgs.os.path, 'exists', return_value=False))

    def worker(self, worker_class=dgs.SweepWorker, *args):
        worker = worker_class(self.station, self.params.copy(), *args)
        worker.points, worker.errors = [], []
        if worker_class is dgs.SweepWorker:
            worker.point_measured.connect(lambda *point: worker.points.append(point), DIRECT)
        worker.error_occurred.connect(worker.errors.append, DIRECT)
        return worker

    def assert_keysight_untouched(self):
        self.assertEqual(self.station.smu.ch1.mock_calls, [])
        self.assertEqual(self.station.smu.ch2.mock_calls, [])
        self.station.smu.enforce_current_compliance.assert_not_called()
        self.station.smu.return_to_local.assert_not_called()

    def test_dac_gates_sweep_without_any_keysight_command(self):
        worker = self.worker()
        worker.run()
        self.assertEqual(worker.errors, [])
        self.assert_keysight_untouched()
        self.assertEqual(len(worker.points), 2)
        for (i, j, c1, c2, v1, v2, spectrum, curr_1, curr_2, meas_v1, meas_v2, *_rest) in worker.points:
            self.assertTrue(math.isnan(curr_1) and math.isnan(curr_2))
            self.assertAlmostEqual(meas_v1, v1)  # DAC setpoint readback
            self.assertAlmostEqual(meas_v2, v2)
        self.assertAlmostEqual(self.dac.dac0.voltage(), .2)
        self.assertAlmostEqual(self.dac.dac3.voltage(), -.3)
        self.assertEqual((self.dac.dac0.ramp_rate(), self.dac.dac3.ramp_rate()), (.2, .3))
        for unused in (1, 2):
            self.assertEqual(self.dac.dac(unused).voltage(), 0.0)

    def test_mixed_outputs_use_only_the_keysight_channel_that_drives_a_gate(self):
        self.params['gate_outputs'] = {1: 'dac:1', 2: 'smu:2'}
        worker = self.worker()
        worker._do_run()
        self.assertEqual(worker.errors, [])
        self.assertEqual(self.station.smu.ch1.mock_calls, [])
        self.station.smu.enforce_current_compliance.assert_not_called()
        self.station.smu.ch2.enforce_current_compliance.assert_called_once_with()
        self.station.smu.ch2.set_r.volt.assert_called_with(-.3)
        self.assertAlmostEqual(self.dac.dac1.voltage(), .2)
        self.assertTrue(all(math.isnan(point[7]) and point[8] == 2e-9 for point in worker.points))

    def test_dac_range_blocks_the_sweep_before_any_write(self):
        self.params.update(c1_end=DAC_RANGE_V + 2)
        worker = self.worker()
        worker._do_run()
        self.assertIn('outside its ±10 V range', ' '.join(worker.errors))
        self.assertEqual(worker.points, [])
        self.assertEqual(self.dac.dac0.voltage(), 0.0)
        self.assert_keysight_untouched()
        self.acquisition.assert_not_called()

    def test_missing_or_shared_dac_blocks_the_sweep(self):
        for outputs, station_dac, message in (({1: 'dac:2', 2: 'dac:2'}, self.dac, 'cannot both'),
                                              ({1: 'dac:0', 2: 'smu:2'}, None, 'not initialized')):
            with self.subTest(message=message):
                self.params['gate_outputs'] = outputs
                self.station.dac = station_dac
                worker = self.worker()
                worker._do_run()
                self.assertIn(message, ' '.join(worker.errors))
                self.assertEqual(worker.points, [])
                self.station.smu.ch2.set_r.volt.assert_not_called()

    def test_final_ramp_returns_dac_gates_to_zero_together(self):
        self.params['ramp_to_zero'] = True
        worker = self.worker()
        targets = []
        worker.voltage_updated.connect(lambda c1, c2, v1, v2: targets.append((v1, v2)), DIRECT)
        worker._do_run()
        self.assertEqual(worker.errors, [])
        self.assertEqual((self.dac.dac0.voltage(), self.dac.dac3.voltage()), (0.0, 0.0))
        final = [pair for pair in targets[2:]]  # After the two sweep points
        self.assertEqual(final[-1], (0.0, 0.0))
        self.assertTrue(all(abs(v1 / .2 - v2 / -.3) < 1e-9 for v1, v2 in final))  # Straight line to 0 V
        self.assert_keysight_untouched()

    def test_pause_and_stop_raise_the_dac_stop_flags(self):
        worker = self.worker()
        worker.pause()
        self.assertTrue(self.dac.dac0.ramp_stop_requested and self.dac.dac3.ramp_stop_requested)
        worker.resume()
        self.assertEqual(worker.prepare_gate_ramp(), worker._cancel_generation)
        self.assertFalse(self.dac.dac0.ramp_stop_requested or self.dac.dac3.ramp_stop_requested)
        worker.stop()
        self.assertTrue(self.dac.dac0.ramp_stop_requested and self.dac.dac3.ramp_stop_requested)
        self.assertEqual(self.station.smu.ch1.mock_calls, [])

    def test_keysight_default_still_checks_both_channels_and_compliance(self):
        del self.params['gate_outputs']
        worker = self.worker()
        worker._do_run()
        self.assertEqual(worker.errors, [])
        self.station.smu.enforce_current_compliance.assert_called_once_with()
        for axis in (1, 2):
            getattr(self.station.smu, f'ch{axis}').source_mode.assert_called()
        self.assertTrue(all(point[7] == 2e-9 and point[8] == 2e-9 for point in worker.points))
        self.assertEqual(self.dac.dac0.voltage(), 0.0)

    def test_position_scan_refuses_uvoltdaq_gates_before_moving(self):
        offsets = {1: 9., 2: 32.}
        self.station.piezo300.get_axis_status.side_effect = lambda axis: {'offset': offsets[axis], 'mode': 'off'}
        self.station.piezo300.get_voltage.side_effect = offsets.__getitem__
        scan = dict(x=np.array([9.]), y=np.array([32.]), move_rate=.5, return_rate=.2, settle=0)
        store = Mock()
        worker = self.worker(PositionSweepWorker, scan, store)
        worker.run()
        self.assertIn('Keysight only', worker.failure)
        self.station.piezo300.set_voltage.assert_not_called()
        self.assertEqual(self.dac.dac0.voltage(), 0.0)
        self.assertEqual(self.station.smu.ch1.set_r.volt.mock_calls, [])


class DgsUVoltDaqDashboardTests(unittest.TestCase):
    setUpClass = classmethod(dashboard_checks.DashboardWorkflowTests.setUpClass.__func__)
    close_dashboard = dashboard_checks.DashboardWorkflowTests.close_dashboard
    wait_for = dashboard_checks.DashboardWorkflowTests.wait_for

    def setUp(self):
        dashboard_checks.DashboardWorkflowTests.setUp(self)
        self.warnings = self.context.enter_context(patch.object(QtWidgets.QMessageBox, "warning"))
        self.window = self.dashboard.dual_window
        self.dashboard.tab_widget.setCurrentIndex(1)

    def select(self, axis, key, window=None):
        combo = getattr(window or self.window, f"ch{axis}_output_combo")
        combo.setCurrentIndex(combo.findData(key))

    def stored(self):
        return json.loads((self.directory / "value_storage").read_text())["dual_gate"]

    def test_selecting_a_dac_output_changes_controls_and_no_voltage(self):
        window = self.window
        self.assertEqual(window.gate_outputs, DEFAULT_GATE_OUTPUTS)
        self.assertTrue(window.ch1_out_btn.isEnabled())
        self.select(1, "dac:0")
        self.assertEqual(window.gate_outputs, {1: "dac:0", 2: "smu:2"})
        self.assertEqual(self.stored()["gate_output_1"], "dac:0")
        self.assertIn("No voltage was changed", window.status_lbl.text())
        self.assertFalse(window.ch1_out_btn.isEnabled())
        self.assertEqual(window.ch1_out_btn.text(), "Always on")
        self.assertFalse(window.ch1_mode.isEnabled())
        self.assertTrue(window.ch1_compliance.isReadOnly())
        self.assertEqual(window.ch1_compliance.specialValueText(), "n/a (uVoltDAQ)")
        self.assertEqual(window.ch1_setpoint.maximum(), DAC_RANGE_V)
        self.assertIn("uVoltDAQ DAC0", window.ch1_box.title())
        self.assertFalse(window.ch1_output_info.isHidden())
        self.assertIn("(no current)", window.right_panel.tabText(window.right_panel.indexOf(window.tab2)))
        # The Keysight gate keeps all of its controls.
        self.assertTrue(window.ch2_out_btn.isEnabled())
        self.assertEqual(window.ch2_setpoint.maximum(), dgs.GATE_MAX_ABS_VOLTAGE_V)
        self.assertEqual(self.station.dac.dac0.voltage(), 0.0)
        self.assertEqual(self.station.smu.ch1.volt(), 0.0)
        # Back to the Keysight restores its controls.
        self.select(1, "smu:1")
        self.assertEqual(window.gate_outputs, DEFAULT_GATE_OUTPUTS)
        self.assertTrue(window.ch1_out_btn.isEnabled())
        self.assertIn(window.ch1_out_btn.text(), ("ON", "OFF"))
        self.assertFalse(window.ch1_compliance.isReadOnly())
        self.assertEqual(window.ch1_setpoint.maximum(), dgs.GATE_MAX_ABS_VOLTAGE_V)
        self.assertEqual(self.stored()["gate_output_1"], "smu:1")
        self.warnings.assert_not_called()
        self.errors.assert_not_called()

    def test_one_dac_output_cannot_drive_both_gates(self):
        self.select(1, "dac:2")
        self.select(2, "dac:2")
        self.warnings.assert_called_once()
        self.assertIn("cannot both", self.warnings.call_args.args[2])
        self.assertEqual(self.window.gate_outputs, {1: "dac:2", 2: "smu:2"})
        self.assertEqual(self.window.ch2_output_combo.currentData(), "smu:2")

    def test_dac_sweep_records_nan_currents_setpoints_and_outputs(self):
        window = self.window
        self.select(1, "dac:0")
        self.select(2, "dac:3")
        for axis in (1, 2):
            getattr(window, f"c{axis}_start").setValue(0)
            getattr(window, f"c{axis}_end").setValue(0.01)
            getattr(window, f"c{axis}_pts").setValue(2)
        window.exposure_box.setValue(0.01)
        window.settling_box.setValue(0)
        window.begin_sweep()
        self.assertFalse(window.ch1_output_combo.isEnabled())
        self.wait_for(lambda: not window.is_running, timeout=10)
        self.assertTrue(np.all(window.data_measured_mask))
        self.assertTrue(np.all(np.isnan(window.data_current_ch1)))
        self.assertFalse(window.t2_heatmap_img.isVisible())
        self.assertIn("no current readback", window.t2_heatmap_win.getPlotItem().titleLabel.text)
        self.assertAlmostEqual(self.station.dac.dac0.voltage(), 0.01)
        self.assertEqual(self.station.smu.ch1.volt(), 0.0)
        self.assertEqual(self.station.smu.ch2.volt(), 0.0)
        self.assertTrue(window.ch1_output_combo.isEnabled())
        self.assertFalse(window.ch1_out_btn.isEnabled())
        files = list(self.directory.rglob("*gate_sweep*.h5"))
        self.assertEqual(len(files), 1)
        with h5py.File(files[0]) as data:
            self.assertEqual(data.attrs["gate_output_ch1"], "dac:0")
            self.assertEqual(data.attrs["gate_output_ch2_label"], "uVoltDAQ DAC3")
            self.assertIn("no current readback", data.attrs["current_ch1_note"])
            self.assertTrue(np.all(np.isnan(data["current_ch1"][()])))
            self.assertTrue(math.isnan(data.attrs["current_compliance_ch1_A"]))
            np.testing.assert_allclose(data["measured_volt_ch1"][()], data["v1_grid"][()], atol=1e-12)
            np.testing.assert_allclose(data["measured_volt_ch2"][()], data["v2_grid"][()], atol=1e-12)
        self.errors.assert_not_called()

    def test_out_of_range_dac_sweep_is_blocked_before_it_starts(self):
        window = self.window
        self.select(1, "dac:0")
        window.c1_start.setValue(0)
        window.c1_end.setValue(DAC_RANGE_V + 2)
        window.begin_sweep()
        self.assertFalse(window.is_running)
        self.errors.assert_called_once()
        self.assertIn("±10 V", self.errors.call_args.args[2])
        self.assertEqual(self.station.dac.dac0.voltage(), 0.0)

    def test_manual_setpoint_ramps_the_selected_dac_output(self):
        window = self.window
        self.select(2, "dac:1")
        window.ch2_setpoint.setValue(0.2)
        window.on_setpoint_enter(2)
        self.wait_for(lambda: not (window.ramp_worker and window.ramp_worker.isRunning()))
        QtWidgets.QApplication.processEvents()
        self.assertAlmostEqual(self.station.dac.dac1.voltage(), 0.2)
        self.assertEqual(self.station.smu.ch2.volt(), 0.0)
        self.assertIn("n/a", window.ch2_live_curr_lbl.text())
        window.read_channel_two()
        self.assertEqual(window.ch2_live_volt_lbl.text(), "Voltage: 0.2000 V")
        self.errors.assert_not_called()
        self.warnings.assert_not_called()

    def test_position_dgs_keeps_the_keysight_outputs(self):
        self.select(1, "dac:0")
        position = self.dashboard.position_dgs_window
        self.assertTrue(position.ch1_output_row.isHidden())
        position.load_storage()
        self.assertEqual(position.gate_outputs, DEFAULT_GATE_OUTPUTS)
        self.assertTrue(position.ch1_out_btn.isEnabled())
        self.assertEqual(self.stored()["gate_output_1"], "dac:0")


if __name__ == "__main__":
    unittest.main()
