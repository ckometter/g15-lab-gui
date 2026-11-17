"""A fake VISA resource that answers like the lab's Keysight B2902B, for offline tests.

It keeps the settings the driver writes and answers queries the way the B2902B does
(numbers in SCPI NR3 form, e.g. ``+1.000000E-08``). A command it does not know is
recorded in ``unexpected`` and raises, so a test can fail on it even when a polling
thread swallows the error.
"""
import logging
import re
from unittest.mock import patch

from qcodes.instrument import Instrument, VisaInstrument

from drivers.keysight_b2902b import KeysightB2902B


class FakeB2902BResource:
    def __init__(self):
        self.commands = []
        self.unexpected = []
        self.settings = {}
        for ch in (1, 2):
            # Power-on state: voltage source at 0 V, output off, 100 uA current limit.
            self.settings.update({f':SOUR{ch}:FUNC:MODE': 'VOLT', f':SOUR{ch}:VOLT': 0.0,
                                  f':SOUR{ch}:CURR': 0.0, f':SENS{ch}:CURR:PROT': 1e-4,
                                  f':SENS{ch}:VOLT:PROT': 2.0, f':OUTP{ch}': 0.0})

    def _reject(self, command):
        self.unexpected.append(command)
        raise AssertionError(f'Unexpected B2902B command: {command!r}')

    def write(self, command):
        self.commands.append(command)
        if command == '*CLS':
            return
        key, _, value = command.partition(' ')
        if key not in self.settings or not value:
            self._reject(command)
        self.settings[key] = value if key.endswith(':FUNC:MODE') else float(value)

    def query(self, command):
        self.commands.append(command)
        if command == '*IDN?':
            return 'Keysight Technologies,B2902B,MY00000000,5.0.2430'
        measured = re.fullmatch(r':MEAS:(VOLT|CURR)\? \(@([12])\)', command)
        if measured:
            quantity, ch = measured.groups()
            voltage = self.settings[f':SOUR{ch}:VOLT'] if self.settings[f':OUTP{ch}'] else 0.0
            return f'{voltage if quantity == "VOLT" else 0.0:+.6E}'
        key = command[:-1]
        if not command.endswith('?') or key not in self.settings:
            self._reject(command)
        value = self.settings[key]
        if isinstance(value, str):
            return value
        return str(int(value)) if key.startswith(':OUTP') else f'{value:+.6E}'

    def clear(self):
        pass

    def close(self):
        pass


def connect_fake_b2902b(name='smu', resource=None):
    """A non-simulated KeysightB2902B whose VISA resource is ``resource``."""
    resource = resource or FakeB2902BResource()

    def open_fake(instrument, name, address, **kwargs):
        Instrument.__init__(instrument, name)
        instrument._visa_handle = resource
        instrument.visa_log = logging.getLogger(f'fake_b2902b.{name}')

    with patch.object(VisaInstrument, '__init__', open_fake), patch.object(KeysightB2902B, 'connect_message'):
        smu = KeysightB2902B(name, 'USB0::FAKE::INSTR', configure_defaults=False)
    return smu, resource
