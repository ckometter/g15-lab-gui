"""Gate caps, ramp timing and cancellation without physical instrument access."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from drivers.keysight_b2902b import B2902BChannel, KeysightB2902B
from gate_safety import (GATE_MAX_ABS_VOLTAGE_V, GATE_MAX_RAMP_STEP_V,
                         GateRampCancelled, validate_gate_targets)
try:
    from tests.test_gate_compliance import FakeSMU
except ModuleNotFoundError:
    from test_gate_compliance import FakeSMU


class GateSafetyTests(unittest.TestCase):
    def setUp(self):
        guard = patch('pyvisa.ResourceManager', side_effect=AssertionError('Physical hardware forbidden'))
        guard.start()
        self.addCleanup(guard.stop)
        # Deterministic settings without reading/writing the user's configuration.
        for name, value in (('_get_ramp_rate', .2), ('_get_ramp_step', .05)):
            setting = patch.object(B2902BChannel, name, return_value=value)
            setting.start()
            self.addCleanup(setting.stop)
        self.smu = FakeSMU()
        self.addCleanup(self.smu.close)
        self.ch = self.smu.ch1
        self.smu.responses[':SOUR1:VOLT?'] = '0'
        wait = patch.object(self.ch, '_wait_for_ramp')
        self.wait = wait.start()
        self.addCleanup(wait.stop)

    def writes(self):
        return [float(command.split(' ', 1)[1]) for command in self.smu.commands
                if command.startswith(':SOUR1:VOLT ')]

    def test_whole_grid_validation_rejects_invalid_transformed_target(self):
        validate_gate_targets(np.array([[-14, 0, 14]]), 0)
        for invalid in (GATE_MAX_ABS_VOLTAGE_V + .0001, -GATE_MAX_ABS_VOLTAGE_V - .0001, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_gate_targets(np.array([[0, 1]]), np.array([[0, invalid]]))
        self.assertEqual(self.smu.commands, [])

    def test_extra_sample_voltage_cap_is_removed(self):
        validate_gate_targets([101, 150, 210], [-101, -150, -210])
        self.assertEqual(self.smu.commands, [])

    def test_invalid_direct_and_ramped_targets_send_no_commands(self):
        for action in (self.ch.volt, self.ch.set_r.volt):
            for invalid in (GATE_MAX_ABS_VOLTAGE_V + .001, -GATE_MAX_ABS_VOLTAGE_V - .001, float('nan'), float('inf')):
                with self.assertRaises(ValueError): action(invalid)
                self.assertEqual(self.smu.commands, [])

    def test_direct_write_uses_fresh_readback_not_qcodes_cache(self):
        self.ch.volt.cache.set(-13)
        self.smu.responses[':SOUR1:VOLT?'] = '.3'
        self.ch.volt(.41)
        points = [.3] + self.writes()
        self.assertEqual(self.smu.commands[0], ':SOUR1:VOLT?')
        self.assertGreater(len(points), 2)
        self.assertLessEqual(max(np.abs(np.diff(points))), GATE_MAX_RAMP_STEP_V + 1e-12)
        self.assertEqual(points[-1], .41)
        self.assertAlmostEqual(sum(call.args[0] for call in self.wait.call_args_list), .11 / .2)

    def test_all_entry_points_ramp_in_both_directions_and_include_short_ramps(self):
        for action in (self.ch.volt, self.ch.set_r.volt):
            for target in (.02, .051, -.13, 0):
                start = float(self.smu.responses[':SOUR1:VOLT?'])
                self.smu.commands.clear()
                self.wait.reset_mock()
                action(target)
                values = self.writes()
                self.assertEqual(len(values), math.ceil(abs(target - start) / .05))
                self.assertLessEqual(max(np.abs(np.diff([start] + values))), .05 + 1e-12)
                self.assertAlmostEqual(sum(c.args[0] for c in self.wait.call_args_list), abs(target-start)/.2)
                self.assertEqual(values[-1], target)

    def test_large_configured_step_still_has_code_only_maximum(self):
        self.ch.ramp_step.get = lambda: 100
        self.ch.set_r.volt(.201)
        self.assertEqual(len(self.writes()), 5)
        self.assertLessEqual(max(np.abs(np.diff([0] + self.writes()))), .05)

    def test_invalid_rate_or_step_cannot_fall_back_to_jump(self):
        for setting in (self.ch.ramp_rate, self.ch.ramp_step):
            for invalid in (0, -1, float('nan'), float('inf')):
                with patch.object(setting, 'get', return_value=invalid):
                    with self.assertRaises(ValueError): self.ch.volt(1)
                self.assertEqual(self.smu.commands, [])

    def test_invalid_ramp_setting_setters_do_not_save(self):
        with patch.object(self.ch, '_save_param_to_storage') as save:
            for setter in (self.ch.ramp_rate, self.ch.ramp_step):
                for value in (0, -1, float('nan'), float('inf')):
                    with self.assertRaises(ValueError): setter(value)
            save.assert_not_called()

    def test_cancel_preserves_intermediate_value_and_does_not_resume_on_retry(self):
        def callback(value): self.ch.ramp_stop_requested = True
        with self.assertRaises(GateRampCancelled): self.ch.set_r.volt(.4, callback)
        self.assertEqual(self.writes(), [.05])
        self.assertAlmostEqual(self.ch.volt.cache.get(), .05)
        self.assertFalse(self.ch.ramping)
        self.smu.commands.clear()
        with self.assertRaises(GateRampCancelled): self.ch.volt(.4)
        self.assertEqual(self.smu.commands, [])

    def test_cancel_while_waiting_prevents_the_pending_step(self):
        self.wait.side_effect = lambda delay: setattr(self.ch, 'ramp_stop_requested', True)
        with self.assertRaises(GateRampCancelled): self.ch.volt(.02)
        self.assertEqual(self.writes(), [])

    def test_qcodes_does_not_add_second_ramp_or_delay(self):
        self.ch._ramp_rate = .2
        self.ch._ramp_step = .05
        self.ch._update_ramping()
        self.assertIsNone(self.ch.volt.step)
        self.assertEqual(self.ch.volt.inter_delay, 0)
        self.ch.volt(.12)
        self.assertEqual(len(self.writes()), 3)

    def test_invalid_initial_readback_blocks_source_write_and_output_enable(self):
        for actual in (str(GATE_MAX_ABS_VOLTAGE_V + .001), str(-GATE_MAX_ABS_VOLTAGE_V - 1), 'nan'):
            self.smu.responses[':SOUR1:VOLT?'] = actual
            for action in (lambda: self.ch.volt(0), lambda: self.ch.output(True),
                           lambda: self.ch.source_mode('VOLT')):
                self.smu.commands.clear()
                with self.assertRaises(ValueError): action()
                self.assertEqual(self.smu.commands, [':SOUR1:VOLT?'])

    def test_output_disable_remains_available_with_invalid_setpoint(self):
        self.smu.responses[':SOUR1:VOLT?'] = '40'
        self.ch.output(False)
        self.assertEqual(self.smu.commands, [':OUTP1 0'])

    def test_output_enable_starts_at_zero_then_ramps_back_to_existing_target(self):
        self.smu.responses[':SOUR1:VOLT?'] = '.12'
        self.ch.output(True)
        enable = self.smu.commands.index(':OUTP1 1')
        before = [float(cmd.split(' ', 1)[1]) for cmd in self.smu.commands[:enable]
                  if cmd.startswith(':SOUR1:VOLT ')]
        after = [float(cmd.split(' ', 1)[1]) for cmd in self.smu.commands[enable + 1:]
                 if cmd.startswith(':SOUR1:VOLT ')]
        self.assertEqual(before[-1], 0)
        self.assertEqual(after[-1], .12)
        self.assertLessEqual(max(np.abs(np.diff([.12] + before + after))), .05)

    def test_cancel_before_enable_keeps_output_off(self):
        self.smu.responses[':SOUR1:VOLT?'] = '.12'
        self.wait.side_effect = lambda delay: setattr(self.ch, 'ramp_stop_requested', True)
        with self.assertRaises(GateRampCancelled): self.ch.output(True)
        self.assertNotIn(':OUTP1 1', self.smu.commands)

    def test_cancel_after_enable_stops_at_intermediate_voltage(self):
        self.smu.responses[':SOUR1:VOLT?'] = '.12'
        def stop_when_enabled(delay):
            if ':OUTP1 1' in self.smu.commands:
                self.ch.ramp_stop_requested = True
        self.wait.side_effect = stop_when_enabled
        with self.assertRaises(GateRampCancelled): self.ch.output(True)
        enable = self.smu.commands.index(':OUTP1 1')
        self.assertFalse(any(c.startswith(':SOUR1:VOLT ') for c in self.smu.commands[enable + 1:]))
        self.assertEqual(self.smu.responses[':SOUR1:VOLT?'], '0.000000000000e+00')

    def test_repeated_enable_does_not_change_an_already_enabled_output(self):
        self.smu.responses[':OUTP1?'] = '1'
        self.smu.responses[':SOUR1:VOLT?'] = '.12'
        self.ch.output(True)
        self.assertEqual(self.writes(), [])
        self.assertNotIn(':OUTP1 1', self.smu.commands)

    def test_emergency_off_cancels_ramp_without_waiting_for_its_lock(self):
        self.wait.side_effect = lambda delay: self.ch.output(False)
        with self.assertRaises(GateRampCancelled): self.ch.volt(.1)
        self.assertIn(':OUTP1 0', self.smu.commands)
        self.assertEqual(self.writes(), [])

    def test_current_mode_cannot_bypass_voltage_cap(self):
        self.smu.responses[':SENS1:VOLT:PROT?'] = str(GATE_MAX_ABS_VOLTAGE_V + 1)
        self.ch.source_mode('CURR')
        self.assertEqual(float(self.smu.responses[':SENS1:VOLT:PROT?']), GATE_MAX_ABS_VOLTAGE_V)
        self.assertLess(self.smu.commands.index(f':SENS1:VOLT:PROT {GATE_MAX_ABS_VOLTAGE_V:.12e}'),
                        self.smu.commands.index(':SOUR1:FUNC:MODE CURR'))
        self.smu.commands.clear()
        with self.assertRaises(ValueError): self.ch.volt_limit(GATE_MAX_ABS_VOLTAGE_V + 1)
        self.assertEqual(self.smu.commands, [])

    def test_live_source_mode_change_is_blocked_to_prevent_voltage_jump(self):
        self.smu.responses[':OUTP1?'] = '1'
        with self.assertRaises(RuntimeError): self.ch.source_mode('CURR')
        self.assertNotIn(':SOUR1:FUNC:MODE CURR', self.smu.commands)

    def test_unconfirmed_final_source_setpoint_fails_ramp(self):
        with patch.object(self.smu, 'write', side_effect=self.smu.commands.append):
            with self.assertRaisesRegex(RuntimeError, 'setpoint not confirmed'):
                self.ch.volt(.1)
        self.assertEqual(self.smu.commands[-1], ':SOUR1:VOLT?')

    def test_voltage_readback_failure_blocks_ramp(self):
        self.smu.responses[':SOUR1:VOLT?'] = RuntimeError('Disconnected')
        with self.assertRaises(RuntimeError): self.ch.volt(1)
        self.assertEqual(self.writes(), [])

    def test_simulation_has_same_bounded_steps_and_cancel_semantics(self):
        smu = KeysightB2902B('ramped_voltage_simulation', 'simulated')
        self.addCleanup(smu.close)
        steps = []
        smu.ch1.set_r.volt(.33, steps.append)
        self.assertEqual(len(steps), 7)
        self.assertLessEqual(max(np.abs(np.diff([0] + steps))), .05)
        self.assertEqual(smu.ch1.volt(), .33)


if __name__ == '__main__': unittest.main()
