"""Verified current compliance; simulated devices and a fake SCPI transport only."""
import threading
import unittest
from unittest.mock import Mock, patch

from qcodes.instrument import Instrument, VisaInstrument
from drivers.keysight_b2902b import B2902BChannel, KeysightB2902B, GATE_CURRENT_COMPLIANCE_A


class FakeSMU(Instrument):
    def __init__(self):
        super().__init__('compliance_transport_test')
        self.simulated = False
        self._lock = threading.RLock()
        self.commands = []
        self.responses = {}
        self.reject_current_compliance = False
        for axis in (1, 2):
            self.responses.update({f':SENS{axis}:CURR:PROT?': '1e-7',
                                   f':SOUR{axis}:CURR?': '0',
                                   f':SOUR{axis}:VOLT?': '3',
                                   f':SENS{axis}:VOLT:PROT?': '10',
                                   f':OUTP{axis}?': '0',
                                   f':SOUR{axis}:FUNC:MODE?': 'VOLT'})
            self.add_submodule(f'ch{axis}', B2902BChannel(self, f'ch{axis}', axis))

    def write(self, cmd):
        self.commands.append(cmd)
        for axis in (1, 2):
            if cmd.startswith(f':SENS{axis}:CURR:PROT ') and not self.reject_current_compliance:
                self.responses[f':SENS{axis}:CURR:PROT?'] = cmd.split(' ', 1)[1]
            for stem in (f':SOUR{axis}:VOLT', f':SENS{axis}:VOLT:PROT', f':OUTP{axis}'):
                if cmd.startswith(stem + ' '):
                    self.responses[stem + '?'] = cmd.split(' ', 1)[1]

    def ask(self, cmd):
        self.commands.append(cmd)
        reply = self.responses[cmd]
        if isinstance(reply, Exception): raise reply
        return reply


class GateComplianceTests(unittest.TestCase):
    def setUp(self):
        self.transport_guard = patch('pyvisa.ResourceManager', side_effect=AssertionError('Physical hardware forbidden in this test'))
        self.transport_guard.start()
        self.addCleanup(self.transport_guard.stop)
        self.ramp_wait = patch.object(B2902BChannel, '_wait_for_ramp')
        self.ramp_wait.start()
        self.addCleanup(self.ramp_wait.stop)

    def simulated(self):
        smu = KeysightB2902B('fixed_compliance_sim_test', 'simulated', configure_defaults=False)
        self.addCleanup(smu.close)
        return smu

    def fake_physical(self):
        smu = FakeSMU()
        self.addCleanup(smu.close)
        return smu

    def test_simulation_accepts_independent_limits_and_ramps_retain_them(self):
        smu = self.simulated()
        for channel, value in ((smu.ch1, 2e-6), (smu.ch2, 5e-5)):
            self.assertEqual(channel.curr_limit(), GATE_CURRENT_COMPLIANCE_A)
            channel.curr_limit(value)
            channel.volt(.01)
            channel.output(True)
            smu.enforce_current_compliance()
            self.assertEqual(channel.curr_limit(), value)
            self.assertEqual(channel.current_compliance_target, value)

    def test_invalid_limits_are_rejected_before_any_command(self):
        smu = self.fake_physical()
        for channel in (smu.ch1, smu.ch2):
            for invalid in (-1, 0, 1e-12, 3.01, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    channel.curr_limit(invalid)
                self.assertEqual(channel.current_compliance_target, GATE_CURRENT_COMPLIANCE_A)
            self.assertEqual(smu.commands, [])

    def test_physical_limits_are_verified_and_retained_on_subsequent_operations(self):
        smu = self.fake_physical()
        for axis, value in ((1, 2e-6), (2, 5e-5)):
            channel = getattr(smu, f'ch{axis}')
            channel.curr_limit(value)
            self.assertEqual(channel.current_compliance_target, value)
            channel.volt(2)
            channel.output(True)
            self.assertEqual(float(smu.responses[f':SENS{axis}:CURR:PROT?']), value)
            self.assertTrue(all(float(cmd.split(' ', 1)[1]) == value for cmd in smu.commands
                                if cmd.startswith(f':SENS{axis}:CURR:PROT ')))

    def test_failed_limit_update_preserves_previous_verified_target(self):
        smu = self.fake_physical()
        smu.reject_current_compliance = True
        with self.assertRaises(RuntimeError):
            smu.ch1.curr_limit(2e-6)
        self.assertEqual(smu.ch1.current_compliance_target, GATE_CURRENT_COMPLIANCE_A)
        self.assertFalse(any(cmd.startswith((':SOUR', ':OUTP')) for cmd in smu.commands))

    def test_voltage_and_output_enable_verify_limit_before_write(self):
        smu = self.fake_physical()
        for axis in (1, 2):
            ch = getattr(smu, f'ch{axis}')
            ch.volt.step = None
            for action, command in ((lambda: ch.volt(2), f':SOUR{axis}:VOLT 2.000000000000e+00'),
                                    (lambda: ch.output(True), f':OUTP{axis} 1')):
                smu.commands.clear()
                action()
                self.assertEqual(smu.commands[:3], [f':SOUR{axis}:VOLT?', f':SENS{axis}:CURR:PROT 1.000000e-07', f':SENS{axis}:CURR:PROT?'])
                self.assertIn(command, smu.commands)
                if command.startswith(':SOUR'):
                    self.assertEqual(smu.commands[-2:], [command, f':SOUR{axis}:VOLT?'])

    def test_failed_or_wrong_readback_blocks_voltage_and_enable(self):
        smu = self.fake_physical()
        smu.reject_current_compliance = True
        for axis in (1, 2):
            ch = getattr(smu, f'ch{axis}')
            ch.volt.step = None
            for response in ('0.1', 'nan', RuntimeError('Disconnected')):
                smu.responses[f':SENS{axis}:CURR:PROT?'] = response
                for action in (lambda: ch.volt(2), lambda: ch.output(True), lambda: ch.source_mode('VOLT')):
                    smu.commands.clear()
                    with self.assertRaises(RuntimeError): action()
                    self.assertFalse(any(cmd.startswith((':SOUR', ':OUTP')) and not cmd.endswith('?') for cmd in smu.commands))

    def test_current_sourcing_cannot_bypass_the_100na_protection(self):
        smu = self.fake_physical()
        for axis in (1, 2):
            ch = getattr(smu, f'ch{axis}')
            for current in (-.001, .001, float('nan')):
                with self.assertRaises(ValueError): ch.curr(current)
            smu.responses[f':SOUR{axis}:CURR?'] = '0.1'
            with self.assertRaises(RuntimeError): ch.source_mode('CURR')
            smu.responses[f':SOUR{axis}:FUNC:MODE?'] = 'CURR'
            with self.assertRaises(RuntimeError): ch.output(True)
            self.assertFalse(any(cmd.startswith(f':OUTP{axis} 1') for cmd in smu.commands))

    def test_reconnect_enforces_both_limits_without_changing_setpoints_or_outputs(self):
        commands = []
        def initialize(instance, name, address, **kwargs):
            Instrument.__init__(instance, name)
            instance._visa_handle = Mock()
        def write(instance, command): commands.append(command)
        def ask(instance, command):
            commands.append(command)
            return '1e-7'
        with patch.object(VisaInstrument, '__init__', initialize), \
             patch.object(VisaInstrument, 'write', write), \
             patch.object(VisaInstrument, 'ask', ask), \
             patch.object(KeysightB2902B, 'connect_message'):
            smu = KeysightB2902B('fixed_compliance_reconnect_test', 'FAKE', configure_defaults=False)
            try:
                for axis in (1, 2):
                    self.assertIn(f':SENS{axis}:CURR:PROT 1.000000e-07', commands)
                    self.assertIn(f':SENS{axis}:CURR:PROT?', commands)
                self.assertFalse(any(cmd.startswith((':SOUR', ':OUTP')) for cmd in commands))
            finally:
                Instrument.close(smu)


if __name__ == '__main__': unittest.main()
