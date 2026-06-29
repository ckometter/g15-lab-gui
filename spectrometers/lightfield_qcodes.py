import Pyro4
from qcodes import Instrument, Parameter
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

class InstrumentSignalBridge(QObject):
    exposure_time_changed = pyqtSignal(float)
    num_frames_changed = pyqtSignal(int)
    file_name_changed = pyqtSignal(str)
    shutter_control_changed = pyqtSignal(int)
    temperature_changed = pyqtSignal(float)
    target_temperature_changed = pyqtSignal(float)
    temperature_status_changed = pyqtSignal(str)
    adc_analog_gain_changed = pyqtSignal(float)
    wavelength_changed = pyqtSignal(float)
    grating_changed = pyqtSignal(object)

lf_signal_bridge = InstrumentSignalBridge()

# Configure Pyro4 to allow pickle serialization
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.PICKLE_PROTOCOL_VERSION = 2

class LightFieldQCoDeSInstrument(Instrument):
    """QCoDeS driver for remote Princeton Instruments LightField.
    Includes a Qt signal bridge (lf_signal_bridge) to emit changes.
    """
    """
    QCoDeS Instrument driver representing a remote Princeton Instruments LightField 
    spectrometer and camera server running via Pyro4.
    """
    # ---------- Helper setter methods that emit Qt signals ----------
    def _set_exposure_time(self, val):
        self.proxy.set_exposure_time(val)
        lf_signal_bridge.exposure_time_changed.emit(val)

    def _set_num_frames(self, val):
        self.proxy.set_num_frames(val)
        lf_signal_bridge.num_frames_changed.emit(val)

    def _set_file_name(self, val):
        self.proxy.set_file_name(val)
        lf_signal_bridge.file_name_changed.emit(val)

    def _set_shutter_control(self, val):
        self.proxy.set_shutter_control(val)
        lf_signal_bridge.shutter_control_changed.emit(val)

    def _set_target_temperature(self, val):
        self.proxy.set_target_temperature(val)
        lf_signal_bridge.target_temperature_changed.emit(val)

    def _set_wavelength(self, val):
        self.proxy.set_wavelength(val)
        lf_signal_bridge.wavelength_changed.emit(val)

    def _set_grating(self, val):
        self.proxy.set_grating(val)
        lf_signal_bridge.grating_changed.emit(val)

    def __init__(self, name: str, host_ip: str, port: int = 50920, **kwargs):
        super().__init__(name, **kwargs)
        
        uri = f"PYRO:LightField@{host_ip}:{port}"
        print(f"Connecting to remote LightField server at {uri}...")
        self.proxy = Pyro4.Proxy(uri)
        
        # 1. Register Camera Parameters
        self.add_parameter(
            'exposure_time',
            label='Exposure Time',
            unit='s',
            get_cmd=lambda: self.proxy.get_exposure_time(),
            set_cmd=self._set_exposure_time
        )
        
        self.add_parameter(
            'num_frames',
            label='Number of Frames to Store',
            get_cmd=lambda: self.proxy.get_num_frames(),
            set_cmd=self._set_num_frames
        )
        
        self.add_parameter(
            'file_name',
            label='Base File Name',
            get_cmd=lambda: self.proxy.get_file_name(),
            set_cmd=self._set_file_name
        )
        
        self.add_parameter(
            'shutter_control',
            label='Shutter Control Mode',
            docstring='1: Normal, 2: Always Closed, 3: Always Open',
            get_cmd=lambda: self.proxy.get_shutter_control(),
            set_cmd=self._set_shutter_control
        )
        
        self.add_parameter(
            'temperature',
            label='Sensor Temperature',
            unit='C',
            get_cmd=lambda: self.proxy.get_temperature()
        )
        
        self.add_parameter(
            'target_temperature',
            label='Target Temperature Set Point',
            unit='C',
            get_cmd=lambda: self.proxy.get_target_temperature(),
            set_cmd=self._set_target_temperature
        )

        self.add_parameter(
            'temperature_status',
            label='Temperature Status',
            get_cmd=lambda: self.proxy.get_temperature_status()
        )

        self.add_parameter(
            'adc_analog_gain',
            label='ADC Analog Gain',
            get_cmd=lambda: self.proxy.get_ADC_analog_gain()
        )

        # 2. Register Spectrometer Parameters
        self.add_parameter(
            'wavelength',
            label='Grating Center Wavelength',
            unit='nm',
            get_cmd=lambda: self.proxy.get_wavelength(),
            set_cmd=self._set_wavelength
        )
        
        self.add_parameter(
            'grating',
            label='Grating Selection',
            get_cmd=lambda: self.proxy.get_grating(),
            set_cmd=self._set_grating
        )

    def get_idn(self):
        """Fetch identification information from remote instrument."""
        try:
            return self.proxy.get_idn()
        except Exception:
            return {"vendor": "Princeton Instruments", "model": "LightField Remote", "serial": None, "firmware": None}

    def initialize_hardware(self):
        """Helper to run the remote initialization sequence."""
        self.proxy.initialize()

    def get_spectrum(self):
        """
        Acquire and return the spectrum as a tuple: (intensities_list, wavelengths_list)
        """
        return self.proxy.get_spectrum()

    def get_spectrum_xarray(self):
        """
        Acquire and return the spectrum as a localized xarray.DataArray.
        (Note: Requires xarray installed on the client machine).
        """
        return self.proxy.get_spectrum_xr()

    def close(self):
        """Release Pyro4 proxy connection when closing the instrument."""
        if hasattr(self, 'proxy'):
            self.proxy._pyroRelease()
        super().close()


# Example Usage (run directly to test)
if __name__ == '__main__':
    # Replace with the actual IP address of the server PC running LightField
    SERVER_IP = "G15-Pylon.dhcp-int.phys.ethz.ch" 
    
    # Initialize the instrument on the client PC
    lf_instrument = LightFieldQCoDeSInstrument(name="lightfield_client", host_ip=SERVER_IP)
    
    print("\nConnected to LightField Instrument!")
    print(f"Current Wavelength: {lf_instrument.wavelength()} {lf_instrument.wavelength.unit}")
    print(f"Current Exposure: {lf_instrument.exposure_time()} {lf_instrument.exposure_time.unit}")
    
    print("\nDisconnecting...")
    lf_instrument.close()
