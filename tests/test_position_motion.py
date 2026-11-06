"""Motion safety with simulated controllers; never access lab hardware."""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PyQt6 import QtTest, QtWidgets

from drivers.attocube_anc150 import AttocubeANC150
from drivers.attocube_anc300 import AttocubeANC300
from measurements.position_tab import PositionTab
from scanner_safety import ScannerSettings


class PositionMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        guard = patch('pyvisa.ResourceManager', side_effect=AssertionError('Hardware access forbidden'))
        guard.start()
        self.addCleanup(guard.stop)
        settings = patch('scanner_safety.ScannerSettings.load', return_value=ScannerSettings())
        settings.start()
        self.addCleanup(settings.stop)
        self.scanner = AttocubeANC300('motion_scanner_test', 'simulated')
        self.stepper = AttocubeANC150('motion_stepper_test', 'simulated')
        self.addCleanup(self.scanner.close)
        self.addCleanup(self.stepper.close)

    def fast_wait(self):
        return patch('scanner_safety.wait_for_motion', side_effect=lambda seconds, keep_running: keep_running())

    def wait_for(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.assertTrue(predicate())
        self.app.processEvents()

    def tab(self):
        with patch.object(PositionTab, 'save_storage'):
            tab = PositionTab(SimpleNamespace(piezo300=self.scanner, piezo150=self.stepper))
        tab.polling_timer.stop()
        self.addCleanup(self.close_tab, tab)
        return tab

    def close_tab(self, tab):
        for worker in (tab.scanner_worker, tab.stepper_worker):
            if worker:
                worker.stop()
                worker.wait(3000)
        self.app.processEvents()
        tab.close()

    def test_every_driver_entry_uses_bounded_waits_before_increments(self):
        events = []
        original = self.scanner._write_voltage_step
        def write(axis, value):
            events.append(('write', axis, value))
            original(axis, value)
        def wait(seconds, keep_running):
            events.append(('wait', seconds))
            return keep_running()
        with patch('scanner_safety.wait_for_motion', side_effect=wait), patch.object(self.scanner, '_write_voltage_step', side_effect=write):
            self.scanner.set_voltage(1, .035)
            self.scanner.voltage_ch2(.02)
            self.scanner._set_voltage(.015)
        previous = {1: 0, 2: 0}
        elapsed = 0
        for index in range(0, len(events), 2):
            self.assertEqual(events[index][0], 'wait')
            _, axis, value = events[index + 1]
            delta = abs(value - previous[axis])
            self.assertLessEqual(delta, .01 + 1e-9)
            self.assertGreaterEqual(events[index][1] + 1e-9, delta / .2)
            elapsed += events[index][1]
            previous[axis] = value
        self.assertAlmostEqual(elapsed, (.035 + .02 + .02) / .2)
        self.assertAlmostEqual(self.scanner.get_voltage(1), .015)

    def test_limits_nonfinite_and_multi_axis_preflight_send_no_writes(self):
        self.scanner.configure_motion(ScannerSettings(maximum=10))
        with patch.object(self.scanner, '_write_voltage_step') as write:
            for value in (-1, 10.1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    self.scanner.ramp_voltages({1: .1, 2: value})
            write.assert_not_called()

    def test_slow_return_from_outside_new_limits_and_cancel_never_jumps(self):
        self.scanner._voltages[1] = 10.02
        self.scanner.configure_motion(ScannerSettings(maximum=10))
        writes = []
        with self.fast_wait(), patch.object(self.scanner, '_write_voltage_step', wraps=self.scanner._write_voltage_step) as write:
            result = self.scanner.ramp_voltages({1: 9.9}, keep_running=lambda: write.call_count < 2)
            writes = [c.args[1] for c in write.call_args_list]
        self.assertFalse(result)
        self.assertEqual(len(writes), 2)
        self.assertGreater(self.scanner.get_voltage(1), 9.9)
        self.assertTrue(all(abs(b-a) <= .01 + 1e-9 for a, b in zip([10.02] + writes, writes)))

    def test_failure_stops_without_retry_and_no_subsequent_axis_move(self):
        with self.fast_wait(), patch.object(self.scanner, '_write_voltage_step', side_effect=RuntimeError('Write failed')) as write:
            with self.assertRaisesRegex(RuntimeError, 'Write failed'):
                self.scanner.ramp_voltages({1: .02, 2: .02})
            self.assertEqual(write.call_count, 1)

    def test_external_change_or_readback_mismatch_stops_ramp(self):
        with self.fast_wait(), patch.object(self.scanner, '_write_voltage_step') as write:
            with self.assertRaisesRegex(RuntimeError, 'readback mismatch'):
                self.scanner.set_voltage(1, .1)
            self.assertEqual(write.call_count, 1)
        with self.fast_wait(), patch.object(self.scanner, 'get_axis_status', side_effect=[
                {'offset': 0, 'mode': 'off'}, {'offset': 1, 'mode': 'off'}]), patch.object(self.scanner, '_write_voltage_step') as write:
            with self.assertRaisesRegex(RuntimeError, 'changed during ramp'):
                self.scanner.set_voltage(1, .1)
            write.assert_not_called()

    def test_settings_and_competing_moves_rejected_while_moving(self):
        with self.scanner.motion_lock:
            with self.assertRaises(RuntimeError): self.scanner.set_voltage(1, .1)
            with self.assertRaises(RuntimeError): self.scanner.configure_motion(ScannerSettings(rate=.1))

    def test_dashboard_setscan_relative_goto_and_stop_remain_responsive(self):
        tab = self.tab()
        tab.scan_command.setText('setscan 0.02, 0.01')
        started = time.monotonic()
        tab.execute_scan_command()
        self.assertLess(time.monotonic() - started, .1)
        self.assertTrue(tab.scanner_stop_btn.isEnabled())
        self.wait_for(lambda: not tab.scanner_moving())
        self.assertGreaterEqual(time.monotonic() - started, .15)
        self.assertAlmostEqual(self.scanner.get_voltage(1), .02)
        tab.scan_command.setText('goto -0.01, 0.02')
        tab.execute_scan_command()
        self.wait_for(lambda: not tab.scanner_moving())
        self.assertAlmostEqual(self.scanner.get_voltage(1), .01)
        self.assertAlmostEqual(self.scanner.get_voltage(2), .03)
        tab.start_scanner_motion({1: 20, 2: 20})
        self.wait_for(lambda: self.scanner.get_voltage(1) > .01)
        tab.scanner_stop_btn.click()
        self.wait_for(lambda: not tab.scanner_moving())
        self.assertLess(self.scanner.get_voltage(1), 1)
        self.assertEqual(self.scanner.get_voltage(2), .03)
        self.assertIn('stopped', tab.scanner_status.text())

    def test_stepper_limits_apply_to_ui_driver_and_existing_device_settings(self):
        tab = self.tab()
        for axis in (1, 2, 3):
            frequency = getattr(self.stepper, f'frequency_ch{axis}')
            amplitude = getattr(self.stepper, f'voltage_ch{axis}')
            frequency(100)
            amplitude(120)
            for value in (101, 0, float('nan')):
                with self.assertRaises(ValueError): frequency(value)
            for value in (120.1, -1, float('inf')):
                with self.assertRaises(ValueError): amplitude(value)
            self.assertEqual(tab.f_edits[axis-1].validator().top(), 100)
            self.assertEqual(tab.a_edits[axis-1].validator().top(), 120)
        self.stepper._frequencies[1] = 200
        with self.assertRaises(ValueError): self.stepper.step(1, 1)
        self.assertEqual(self.stepper.pseudo_position(1), 0)

    def test_stepper_goto_zero_and_relative_coordinates(self):
        tab = self.tab()
        tab.start_stepper_motion(targets={1: 3, 2: -2, 3: 1})
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertEqual([self.stepper.pseudo_position(a) for a in (1, 2, 3)], [3, -2, 1])
        physical = self.stepper._positions.copy()
        tab.zero_stepper_coordinates()
        self.assertEqual(self.stepper._positions, physical)
        tab.start_stepper_motion(targets={1: -1, 2: 2}, relative=True)
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertEqual([self.stepper.pseudo_position(a) for a in (1, 2, 3)], [-1, 2, 0])
        tab.start_stepper_motion(targets={1: 0, 2: 0, 3: 0})
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertEqual(self.stepper._positions, physical)

    def test_stepper_jog_stop_counts_completed_commands(self):
        tab = self.tab()
        tab.start_stepper_motion(jog=(1, 1))
        self.wait_for(lambda: self.stepper.pseudo_position(1) >= 2)
        tab.stop_stepper_motion()
        self.wait_for(lambda: not tab.stepper_moving())
        count = self.stepper.pseudo_position(1)
        QtTest.QTest.qWait(50)
        self.assertEqual(self.stepper.pseudo_position(1), count)
        self.assertEqual(count, self.stepper._positions[1])

    def test_stepper_uncertain_write_invalidates_coordinate(self):
        self.stepper.simulated = False
        self.addCleanup(setattr, self.stepper, 'simulated', True)
        with patch.object(self.stepper, 'validate_drive', return_value=100), patch.object(self.stepper, 'write', side_effect=RuntimeError('Disconnected')):
            with self.assertRaises(RuntimeError): self.stepper.step(1, 1)
        self.assertIsNone(self.stepper.pseudo_position(1))
        self.stepper.zero_pseudo_position()
        self.assertEqual(self.stepper.pseudo_position(1), 0)

    def test_bad_saved_settings_block_motion_until_explicitly_repaired(self):
        with patch('scanner_safety.ScannerSettings.load', side_effect=ValueError('bad limits')):
            self.scanner.init_motion_safety()
        with patch.object(self.scanner, '_write_voltage_step') as write:
            with self.assertRaisesRegex(RuntimeError, 'Invalid saved'):
                self.scanner.set_voltage(1, .1)
            write.assert_not_called()
        self.scanner.configure_motion(ScannerSettings())
        with self.fast_wait(): self.scanner.set_voltage(1, .01)

    def test_saved_settings_round_trip_without_any_motion(self):
        tab = self.tab()
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / 'value_storage'
            storage.write_text(json.dumps({'shared': {'keep': 123}, 'position': {'vx': 45}}))
            with patch('measurements.position_tab.parent_dir', directory), patch.object(self.scanner, '_write_voltage_step') as write:
                for key, value in dict(minimum=2, maximum=12, rate=.1, increment=.005).items():
                    tab.scanner_setting_inputs[key].setValue(value)
                tab.apply_scanner_settings()
                saved = json.loads(storage.read_text())
                self.assertEqual(saved['shared'], {'keep': 123})
                self.assertEqual(saved['position']['scanner_safety'], dict(minimum=2, maximum=12, rate=.1, increment=.005))
                tab.load_storage()
                write.assert_not_called()
                self.assertEqual(self.scanner.scanner_settings.maximum, 12)

    def test_routine_save_does_not_replace_invalid_limits_with_defaults(self):
        tab = self.tab()
        self.scanner.scanner_settings_error = 'Invalid saved limits'
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / 'value_storage'
            invalid = {'maximum': -1}
            storage.write_text(json.dumps({'position': {'scanner_safety': invalid}}))
            with patch('measurements.position_tab.parent_dir', directory):
                tab.save_storage()
                self.assertEqual(json.loads(storage.read_text())['position']['scanner_safety'], invalid)
                self.assertTrue(self.scanner.scanner_settings_error)
                tab.apply_scanner_settings()
                self.assertFalse(self.scanner.scanner_settings_error)
                self.assertEqual(json.loads(storage.read_text())['position']['scanner_safety']['maximum'], 60)

    def test_maps_reject_entire_grid_before_moving_and_propagate_write_failure(self):
        from measurements.scalar_map_gui import ScalarMapWorker
        from measurements.photoluminescence_map_gui import PLMapWorker
        params = dict(x_param='position X', y_param='position Y', x_start=0, x_end=.1,
                      x_pts=2, y_start=0, y_end=1, y_pts=2)
        self.scanner.configure_motion(ScannerSettings(maximum=.5))
        station = SimpleNamespace(piezo300=self.scanner)
        for cls in (ScalarMapWorker, PLMapWorker):
            with self.subTest(worker=cls.__name__):
                worker = cls(station, params)
                with patch.object(self.scanner, '_write_voltage_step') as write:
                    with self.assertRaises(ValueError): worker.preflight_gate_scan([0, .1], [0, 1])
                    write.assert_not_called()
                with self.fast_wait(), patch.object(self.scanner, '_write_voltage_step', side_effect=RuntimeError('Readback failed')) as write:
                    with self.assertRaisesRegex(RuntimeError, 'Readback failed'): worker.set_piezo(1, .1)
                    self.assertEqual(write.call_count, 1)
                worker.stop()
                with patch.object(self.scanner, '_write_voltage_step') as write:
                    from gate_safety import GateRampCancelled
                    with self.assertRaises(GateRampCancelled): worker.set_piezo(1, .1)
                    write.assert_not_called()

    def test_native_offset_write_is_ramped_and_bad_axis_mode_blocks_whole_move(self):
        # Exercise the physical driver's public command routing using fake I/O.
        writes = []
        def exchange(command):
            op, axis, value = command.split()
            self.assertEqual(op, 'seta')
            writes.append(float(value))
            self.scanner._voltages[int(axis)] = float(value)
        if not hasattr(self.scanner, 'write_raw'):
            self.skipTest('Physical ANC300 package unavailable')
        with self.fast_wait(), patch.object(self.scanner, '_exchange', side_effect=exchange), \
             patch.object(self.scanner, 'simulated', False), \
             patch.object(self.scanner, 'get_axis_status', side_effect=lambda a: {'offset': self.scanner._voltages[a], 'mode': 'off'}), \
             patch.object(self.scanner, 'get_voltage', side_effect=lambda a: self.scanner._voltages[a]):
            self.scanner.write_raw('seta 1 0.025')
        self.assertAlmostEqual(self.scanner.get_voltage(1), .025)
        self.assertGreater(len(writes), 1)
        with patch.object(self.scanner, 'get_axis_status', side_effect=[
                {'offset': .025, 'mode': 'off'}, {'offset': 0, 'mode': 'gnd'}]), patch.object(self.scanner, '_write_voltage_step') as write:
            with self.assertRaises(RuntimeError): self.scanner.ramp_voltages({1: .1, 2: .1})
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
