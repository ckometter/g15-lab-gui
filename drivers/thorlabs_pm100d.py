import time
import random
import numpy as np
from qcodes.instrument import Instrument
from qcodes_contrib_drivers.drivers.Thorlabs.PM100D import Thorlab_PM100D

class ThorlabsPM100D(Thorlab_PM100D):
    """
    Wrapper for Thorlab_PM100D that supports safe simulation fallback and faster readout.
    """
    def __init__(self, name: str, address: str, **kwargs):
        self.simulated = (address == 'simulated')
        if self.simulated:
            # Bypass VisaInstrument and Thorlab_PM100D base __init__
            # to avoid opening a non-existent VISA resource.
            Instrument.__init__(self, name, **kwargs)
            self._power = 1.2e-6
            self._wavelength = 532e-9
            self._averaging = 300
            self._auto_range = 1
            self._attenuation = 0.0
            self._power_range = 1e-3
            self._frequency = 0.0
            self._current = 1.2e-9
            self._current_range = 1e-6
            self._zero_value = 0.0
            self._beam_diameter = 1.0

            self.add_parameter('power',
                               label='Measured power',
                               unit='W',
                               get_cmd=self._get_simulated_power)
            
            self.add_parameter('wavelength',
                               label='Detected wavelength',
                               unit='m',
                               get_cmd=lambda: self._wavelength,
                               set_cmd=self._set_simulated_wavelength)
            
            self.add_parameter('averaging',
                               label='Averaging rate',
                               get_cmd=lambda: self._averaging,
                               set_cmd=self._set_simulated_averaging)
            
            self.add_parameter('auto_range',
                               label='Auto range power',
                               get_cmd=lambda: self._auto_range,
                               set_cmd=self._set_simulated_auto_range)

            self.add_parameter('attenuation',
                               label='Attenuation',
                               unit='dB',
                               get_cmd=lambda: self._attenuation,
                               set_cmd=self._set_simulated_attenuation)

            self.add_parameter('power_range',
                               label='Power range',
                               unit='W',
                               get_cmd=lambda: self._power_range,
                               set_cmd=self._set_simulated_power_range)

            self.add_parameter('frequency',
                               unit='Hz',
                               get_cmd=lambda: self._frequency)

            self.add_parameter('current',
                               label='Current',
                               unit='A',
                               get_cmd=self._get_simulated_current)

            self.add_parameter('current_range',
                               label='Current range',
                               unit='A',
                               get_cmd=lambda: self._current_range)

            self.add_parameter('zero_value',
                               unit='W',
                               get_cmd=lambda: self._zero_value)

            self.add_parameter('beam_diameter',
                               label='Beam diameter',
                               unit='m',
                               get_cmd=lambda: self._beam_diameter,
                               set_cmd=self._set_simulated_beam_diameter)
        else:
            # Initialize physical VISA connection
            super().__init__(name, address, **kwargs)

    def _get_power(self) -> float:
        """
        Overridden to speed up power measurement of physical device.
        Avoids redundant configuration commands and sleeps by calling READ? directly.
        """
        if self.simulated:
            return self._get_simulated_power()
        return float(self.ask('READ?'))

    def _get_simulated_power(self) -> float:
        # Generates a small simulated power value around 1.2 uW with a sine fluctuation
        t = time.time()
        base = 1.2e-5 + 4e-6 * np.sin(t / 5.0)
        noise = random.normalvariate(0, 1e-7)
        return float(max(1e-9, base + noise))

    def _set_simulated_wavelength(self, val: float):
        self._wavelength = val

    def _set_simulated_averaging(self, val: int):
        self._averaging = val

    def _set_simulated_auto_range(self, val: int):
        self._auto_range = val

    def _set_simulated_attenuation(self, val: float):
        self._attenuation = val

    def _set_simulated_power_range(self, val: float):
        self._power_range = val

    def _set_simulated_beam_diameter(self, val: float):
        self._beam_diameter = val

    def _get_simulated_current(self) -> float:
        # Returns simulated photo-current proportional to power
        return self._get_simulated_power() * 0.5

    def get_idn(self) -> dict:
        if self.simulated:
            return {'vendor': 'Simulated', 'model': 'Simulated',
                    'serial': '1234', 'firmware': '1.0'}
        return super().get_idn()
