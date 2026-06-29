import time
import random
import numpy as np
from qcodes.instrument import Instrument
from qcodes_contrib_drivers.drivers.Thorlabs.PM100D import Thorlab_PM100D

class ThorlabsPM100D(Thorlab_PM100D):
    """
    Wrapper for Thorlab_PM100D that supports safe simulation fallback.
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
        else:
            # Initialize physical VISA connection
            super().__init__(name, address, **kwargs)

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

    def get_idn(self) -> dict:
        if self.simulated:
            return {'vendor': 'Simulated', 'model': 'Simulated',
                    'serial': '1234', 'firmware': '1.0'}
        return super().get_idn()
