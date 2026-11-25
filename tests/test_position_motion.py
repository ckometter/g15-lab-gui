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

from PyQt6 import QtCore, QtTest, QtWidgets

from drivers.attocube_anc150 import AttocubeANC150
from drivers.attocube_anc300 import AttocubeANC300
from fake_anc150 import FakeANC150Port, connect_fake
from measurements.position_tab import PositionTab
from scanner_safety import MAX_INCREMENT, MAX_RAMP_RATE, ScannerSettings


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

    def test_increment_above_old_cap_sets_the_step_even_at_full_rate(self):
        # Until 18 Nov 2026 a step was also capped at 0.05 s of travel, so at 1 V/s
        # every ramp crawled in 0.05 V steps whatever increment was applied.
        events = []
        original = self.scanner._write_voltage_step
        def write(axis, value):
            events.append(('write', value))
            original(axis, value)
        def wait(seconds, keep_running):
            events.append(('wait', seconds))
            return keep_running()
        self.scanner._voltages[1] = 32.099
        self.scanner.configure_motion(ScannerSettings(rate=1, increment=.5))
        with patch('scanner_safety.wait_for_motion', side_effect=wait), \
             patch.object(self.scanner, '_write_voltage_step', side_effect=write):
            self.assertTrue(self.scanner.ramp_voltages({1: 33.299}))
            self.assertEqual([kind for kind, _ in events], ['wait', 'write'] * 3)
            for (_, seconds), (_, value), expected in zip(events[::2], events[1::2], (32.499, 32.899, 33.299)):
                self.assertAlmostEqual(seconds, .4)
                self.assertAlmostEqual(value, expected)
            # 33.299 + 0.1 differs from 33.299 by slightly more than 0.1 V: still one step.
            events.clear()
            self.scanner.configure_motion(ScannerSettings(rate=1, increment=.1))
            self.assertTrue(self.scanner.ramp_voltages({1: .1}, relative=True))
            self.assertEqual([kind for kind, _ in events], ['wait', 'write'])
            self.assertAlmostEqual(events[0][1], .1)
        self.assertAlmostEqual(self.scanner.get_voltage(1), 33.399)

    def test_dashboard_accepts_typed_rate_and_increment_above_the_old_caps(self):
        # Until 18 Nov 2026 these boxes stopped at 1 V/s and 0.05 V.
        tab = self.tab()
        boxes = tab.scanner_setting_inputs
        self.assertEqual(boxes['rate'].maximum(), MAX_RAMP_RATE)
        self.assertEqual(boxes['increment'].maximum(), MAX_INCREMENT)
        with tempfile.TemporaryDirectory() as directory:
            with patch('measurements.position_tab.parent_dir', directory), \
                 patch.object(self.scanner, '_write_voltage_step') as write:
                for key, text in (('rate', '5'), ('increment', '0.5')):
                    boxes[key].selectAll()
                    QtTest.QTest.keyClicks(boxes[key], text)
                    QtTest.QTest.keyClick(boxes[key], QtCore.Qt.Key.Key_Return)
                tab.apply_scanner_settings()
                saved = json.loads((Path(directory) / 'value_storage').read_text())['position']['scanner_safety']
            write.assert_not_called()
        self.assertEqual((saved['rate'], saved['increment']), (5, .5))
        settings = self.scanner.scanner_settings
        self.assertEqual((settings.rate, settings.increment), (5, .5))
        waits = []
        with patch('scanner_safety.wait_for_motion', side_effect=lambda seconds, keep_running: waits.append(seconds) or True):
            self.assertTrue(self.scanner.ramp_voltages({1: 1}))
        self.assertEqual(len(waits), 2)  # two 0.5 V steps, each after 0.1 s at 5 V/s
        for seconds in waits:
            self.assertAlmostEqual(seconds, .1)
        self.assertAlmostEqual(self.scanner.get_voltage(1), 1)

    def test_apply_button_is_marked_while_typed_settings_are_not_applied(self):
        # On 18 Nov 2026 typed values looked applied, but ramps kept using 1 V/s.
        tab = self.tab()
        button, rate = tab.apply_scanner_settings_btn, tab.scanner_setting_inputs['rate']
        def type_text(box, text, enter=True):
            box.selectAll()
            QtTest.QTest.keyClicks(box, text)
            if enter:
                QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_Return)
        self.assertEqual(button.text(), 'Apply and save safety settings')
        self.assertIn('at most 0.2 V/s', button.toolTip())
        type_text(rate, '5', enter=False)
        self.assertEqual(button.text(), 'Apply and save safety settings (not applied yet)')  # while typing
        QtTest.QTest.keyClick(rate, QtCore.Qt.Key.Key_Return)
        self.assertIn('not applied yet', button.text())
        self.assertIn('color: #b91c1c', button.styleSheet())
        self.assertEqual(self.scanner.scanner_settings.rate, .2)  # ramps still use the old rate
        with tempfile.TemporaryDirectory() as directory, patch('measurements.position_tab.parent_dir', directory):
            tab.apply_scanner_settings()
            self.assertEqual(self.scanner.scanner_settings.rate, 5)
            self.assertEqual((button.text(), button.styleSheet()), ('Apply and save safety settings', ''))
            self.assertIn('at most 5 V/s', button.toolTip())
            type_text(rate, '5')  # the applied value typed again: unmarked after Enter
            self.assertNotIn('not applied', button.text())
            type_text(tab.scanner_setting_inputs['minimum'], '70')  # above the maximum: Apply refuses it
            tab.apply_scanner_settings()
            self.assertIn('not applied yet', button.text())
            self.assertEqual(self.scanner.scanner_settings.minimum, 0)
        tab.sync_scanner_settings()  # reloading the tab shows the applied values again
        self.assertEqual(tab.scanner_setting_inputs['minimum'].value(), 0)
        self.assertNotIn('not applied', button.text())

    def test_settings_accept_rates_and_increments_up_to_their_caps_only(self):
        settings = ScannerSettings(rate=MAX_RAMP_RATE, increment=MAX_INCREMENT)
        self.assertEqual((settings.rate, settings.increment), (MAX_RAMP_RATE, MAX_INCREMENT))
        with self.assertRaisesRegex(ValueError, f'0.001 to {MAX_RAMP_RATE:g} V/s'):
            ScannerSettings(rate=MAX_RAMP_RATE + .001)
        with self.assertRaisesRegex(ValueError, f'0.001 to {MAX_INCREMENT:g} V'):
            ScannerSettings(increment=MAX_INCREMENT + .001)

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
            with self.assertRaisesRegex(RuntimeError, r'readback mismatch: requested 0\.0100 V, '
                                                      r'read back 0\.0000 V, then 0\.0000 V'):
                self.scanner.set_voltage(1, .1)
            self.assertEqual(write.call_count, 1)
        # The check before each step reads the offset and mode; a change seen twice stops the ramp.
        with self.fast_wait(), patch.object(self.scanner, 'get_axis_status', return_value={'offset': 0, 'mode': 'off'}), \
             patch.object(self.scanner, 'get_offset_and_mode', return_value={'offset': 1, 'mode': 'off'}) as check, \
             patch.object(self.scanner, '_write_voltage_step') as write:
            with self.assertRaisesRegex(RuntimeError, r"changed during ramp: expected 0\.0000 V in offset mode 'off', "
                                                      r"read 1\.0000 V in mode 'off', then 1\.0000 V in mode 'off'"):
                self.scanner.set_voltage(1, .1)
            write.assert_not_called()
            self.assertEqual(check.call_count, 2)

    def test_one_stray_readback_is_read_again_but_a_wrong_offset_still_stops(self):
        # On 18 Nov 2026 a position scan stopped on a single readback mismatch.
        self.scanner._voltages[1] = 8.0
        self.scanner.configure_motion(ScannerSettings(rate=1, increment=1))
        real = self.scanner.get_voltage
        replies = iter([None, None, 9.004])  # the third read is the first step's readback
        def stray_once(axis):
            reply = next(replies, None)
            return real(axis) if reply is None else reply
        with self.fast_wait(), patch.object(self.scanner, 'get_voltage', side_effect=stray_once), \
             patch('builtins.print') as log:
            self.assertTrue(self.scanner.ramp_voltages({1: 10}))
        self.assertEqual(self.scanner._voltages[1], 10)
        log.assert_called_once()
        self.assertIn('read back 9.0040 V after requesting 9.0000 V; a second read gave 9.0000 V',
                      log.call_args.args[0])
        # An offset that reads wrong twice still stops the ramp before any further step.
        wrong_at_11 = lambda axis: 10.5 if self.scanner._voltages[1] == 11 else real(axis)
        with self.fast_wait(), patch.object(self.scanner, 'get_voltage', side_effect=wrong_at_11), \
             patch.object(self.scanner, '_write_voltage_step', wraps=self.scanner._write_voltage_step) as write:
            with self.assertRaisesRegex(RuntimeError, r'requested 11\.0000 V, read back 10\.5000 V, then 10\.5000 V'):
                self.scanner.ramp_voltages({1: 12})
            self.assertEqual(write.call_count, 1)

    def test_each_ramp_step_sends_four_commands(self):
        # Until 18 Nov 2026 every step sent 9: the driver repeated the ramp's checks.
        if not hasattr(self.scanner, 'write_raw'):
            self.skipTest('Physical ANC300 package unavailable')
        offsets, sent = {1: 8.0}, []
        def exchange(command):
            sent.append(command)
            op, axis, *value = command.split()
            if op == 'seta':
                offsets[int(axis)] = float(value[0])
                return ''
            return {'geta': f'{offsets[int(axis)]:.6f}', 'geto': f'{offsets[int(axis)]:.6f}', 'getm': 'off'}[op]
        self.scanner.configure_motion(ScannerSettings(rate=1, increment=1))
        with self.fast_wait(), patch.object(self.scanner, 'simulated', False), \
             patch.object(self.scanner, '_exchange', side_effect=exchange):
            self.assertTrue(self.scanner.ramp_voltages({1: 10}))
        self.assertEqual(sent, ['geta 1', 'geto 1', 'getm 1',                      # before any motion
                                'geta 1', 'getm 1', 'seta 1 9.000000', 'geta 1',  # step to 9 V
                                'geta 1', 'getm 1', 'seta 1 10.000000', 'geta 1'])
        self.assertEqual(offsets[1], 10)

    def test_dashboard_does_not_poll_the_scanner_while_a_ramp_holds_it(self):
        tab = self.tab()
        with patch.object(self.scanner, 'get_axis_status', wraps=self.scanner.get_axis_status) as status:
            with self.scanner.motion_lock:
                tab.poll_voltages()
            status.assert_not_called()
            tab.poll_voltages()
            self.assertEqual(status.call_count, 2)  # both axes again once the ramp is over

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

    def fake_port(self, **settings):
        """Talk to an offline stand-in for the lab's ANC150 through the driver's real serial code."""
        port = FakeANC150Port(**settings)
        connect_fake(self.stepper, port)
        self.addCleanup(setattr, self.stepper, 'simulated', True)
        return port

    def test_held_arrow_steps_near_the_set_frequency_in_counted_bursts(self):
        port = self.fake_port(frequencies=(100, 43, 52), latency=0.01)
        tab = self.tab()
        tab.start_stepper_motion(jog=(2, 1))
        QtTest.QTest.qWait(1000)
        tab.stop_stepper_motion()
        self.wait_for(lambda: not tab.stepper_moving())
        # The front panel makes 43 steps/s; one command per single step made ~12 here (~6 at the lab's 25 ms).
        self.assertGreater(port.steps[2], 0.6 * 43)
        self.assertEqual(self.stepper.pseudo_position(2), port.steps[2])
        self.assertEqual(port.overlaps, 0)  # never a step command while the axis is still stepping
        self.assertEqual(port.mode[2], 'stp')

    def test_click_is_one_step_and_goto_sends_counted_bursts(self):
        port = self.fake_port(frequencies=(100, 100, 52), latency=0.005)
        tab = self.tab()
        tab.move_stepper('x', -1)
        self.wait_for(lambda: not tab.stepper_moving())
        tab.start_stepper_motion(targets={2: 60}, relative=True)
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertEqual([c for c in port.commands if c.startswith('step')],
                         ['stepd 1 1', 'stepu 2 25', 'stepu 2 25', 'stepu 2 10'])
        self.assertEqual((port.steps[1], port.steps[2], port.overlaps), (-1, 60, 0))
        self.assertEqual([self.stepper.pseudo_position(a) for a in (1, 2)], [-1, 60])
        self.assertEqual(tab.stepper_status.text(), 'Stepper move complete.')

    def test_setting_above_the_limit_names_the_axis_and_value_and_sends_no_motion(self):
        port = self.fake_port(frequencies=(102, 43, 52))
        tab = self.tab()
        tab.poll_voltages()
        self.assertEqual(tab.sf1_label.text(), 'Fx: 102 Hz (max 100)')
        self.assertIn('X will not step', tab.sf1_label.toolTip())
        self.assertEqual((tab.sf2_label.text(), tab.sf2_label.toolTip()), ('Fy: 43 Hz', ''))
        tab.move_stepper('x', 1)
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertFalse([c for c in port.commands if c.startswith(('setm', 'step'))])
        self.assertTrue(tab.stepper_status.text().startswith('X: ANC150 axis 1 is set to 102 Hz'))
        self.assertIn('color', tab.stepper_status.styleSheet())
        port.voltage[3] = 130
        tab.start_stepper_motion(targets={2: 1, 3: 1}, relative=True)
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertEqual(port.steps, {1: 0, 2: 0, 3: 0})  # both axes are checked before either moves
        self.assertIn('Z: ANC150 axis 3 is set to 130 V', tab.stepper_status.text())

    def test_arrows_move_with_an_unknown_coordinate_but_absolute_goto_needs_zero(self):
        tab = self.tab()
        self.stepper._pseudo_positions[1] = None
        tab.move_stepper('x', 1)
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertEqual((self.stepper._positions[1], self.stepper.pseudo_position(1)), (1, None))
        tab.start_stepper_motion(targets={1: 5})
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertEqual(self.stepper._positions[1], 1)
        self.assertIn('Set zero before an absolute Goto', tab.stepper_status.text())

    def test_simulated_stepper_says_nothing_moves(self):
        tab = self.tab()
        self.assertIn('SIMULATED, nothing moves', tab.stepper_group.title())
        tab.move_stepper('z', 1)
        self.wait_for(lambda: not tab.stepper_moving())
        self.assertIn('no hardware moved', tab.stepper_status.text())
        self.fake_port()
        tab.refresh_stepper_coordinates()
        self.assertEqual(tab.stepper_group.title(), 'Stepper (ANC150)')

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
            op, axis, *value = command.split()
            if op == 'getm':  # the mode check before each step
                return 'off'
            self.assertEqual(op, 'seta')
            writes.append(float(value[0]))
            self.scanner._voltages[int(axis)] = float(value[0])
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
