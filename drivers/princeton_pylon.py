import numpy as np
try:
    import pylablib as pll
    from pylablib.devices import PrincetonInstruments as pi
except ImportError:
    pll = None

from qcodes.instrument import Instrument
from qcodes.parameters import ArrayParameter

class CCDImage(ArrayParameter):
    """
    QCoDeS ArrayParameter representing a 2D frame read from the CCD.
    """
    def __init__(self, name, instrument, shape=(512, 512), **kwargs):
        setpoints = (np.arange(shape[0]), np.arange(shape[1]))
        super().__init__(name=name,
                         shape=shape,
                         instrument=instrument,
                         unit='counts',
                         label='CCD Image Frame',
                         setpoint_names=('y', 'x'),
                         setpoint_labels=('Row pixel', 'Column pixel'),
                         setpoint_units=('px', 'px'),
                         setpoints=setpoints,
                         **kwargs)
        
    def get_raw(self) -> np.ndarray:
        return self.instrument.capture_image()


class PrincetonPylon(Instrument):
    """
    QCoDeS driver for the Princeton Instruments PyLoN CCD Camera.
    Uses pylablib's PICam module under the hood.
    
    In simulation mode, it should just simulate a gaussian.
    """
    def __init__(self, name, camera_id=None, **kwargs):
        # Remove and close any existing instrument with the same name to avoid collision
        try:
            instr = Instrument.find_instrument(name)
            try:
                Instrument.remove_instance(instr)
            except Exception:
                pass
            try:
                instr.close()
            except Exception:
                pass
        except KeyError:
            pass

        # We inherit from base Instrument as pylablib handles low-level drivers.
        super().__init__(name, **kwargs)
        
        self.simulated = (camera_id is None or camera_id == 'simulated')
        self.camera_id = camera_id
        self._exposure_time = 0.1 # default 100ms
        self._sensor_shape = (512, 512)
        
        if self.simulated:
            print(f"Initialized Princeton PyLoN '{name}' in Simulation Mode.")
        else:
            if pll is None:
                raise ImportError(
                    "The 'pylablib' library is required to control physical Princeton cameras. "
                    "Run 'pip install pylablib' to install it."
                )
            # Connect to physical camera
            self.cam = pi.PicamCamera(camera_idx=camera_id)
            # Synchronize shape with physical sensor size
            detector_size = self.cam.get_detector_size() # returns (width, height)
            self._sensor_shape = (detector_size[1], detector_size[0]) # (rows, cols)
            
        self._source_wavelength = None
        self._source_sigma = None

        # Add parameter for exposure time
        self.add_parameter('exposure_time',
                           label='Exposure Time',
                           unit='s',
                           get_cmd=self._get_exposure_time,
                           set_cmd=self._set_exposure_time)
        
        # Add parameter for CCD image
        self.add_parameter('image',
                           parameter_class=CCDImage,
                           shape=self._sensor_shape)

        # Add parameter for source wavelength (for simulation mode alignment)
        self.add_parameter('source_wavelength',
                           label='Source Wavelength',
                           unit='nm',
                           get_cmd=self._get_source_wavelength,
                           set_cmd=self._set_source_wavelength)

        # Add parameter for source sigma (simulated Gaussian spectral width)
        self.add_parameter('source_sigma',
                           label='Source Sigma',
                           unit='nm',
                           get_cmd=self._get_source_sigma,
                           set_cmd=self._set_source_sigma)

    def close(self) -> None:
        if getattr(self, 'simulated', False):
            Instrument.close(self)
        else:
            try:
                super().close()
            except Exception:
                self.remove_instance(self)

    def get_idn(self) -> dict:
        return {
            'vendor': 'Princeton Instruments',
            'model': 'PyLoN CCD',
            'serial': 'simulated' if self.simulated else str(self.camera_id),
            'firmware': None
        }


    def _get_exposure_time(self) -> float:
        if self.simulated:
            return self._exposure_time
        return float(self.cam.get_exposure())

    def _set_exposure_time(self, val: float) -> None:
        if self.simulated:
            self._exposure_time = val
        else:
            self.cam.set_exposure(val)

    def _get_source_wavelength(self) -> float:
        if self._source_wavelength is None:
            return 490.0
        return self._source_wavelength


    def _set_source_wavelength(self, val: float) -> None:
        self._source_wavelength = val

    def _get_source_sigma(self) -> float:
        if self._source_sigma is None:
            return 1.25
        return self._source_sigma

    def _set_source_sigma(self, val: float) -> None:
        self._source_sigma = val

    def capture_image(self) -> np.ndarray:
        """Capture a single image frame."""
        if self.simulated:
            # Generate a realistic 2D Gaussian laser spot with noise
            shape = self._sensor_shape
            
            # We can retrieve other instruments dynamically from the QCoDeS station
            # to make the simulated spot shift with the monochromator wavelength!
            center_x, center_y = shape[1] // 2, shape[0] // 2
            
            # Try to grab wavelength from monochromator to shift the spot based on dispersion
            try:
                monochromator = Instrument.find_instrument('monochromator')
                wl_mono = float(monochromator.wavelength())
                wl_source = float(self.source_wavelength())
                # Shift spot horizontally based on difference (0.05 nm/pixel dispersion)
                # If monochromator wavelength is larger than source, peak shifts to left (lower pixel index)
                dispersion = 0.05
                center_x = shape[1] // 2 + int((wl_source - wl_mono) / dispersion)
            except Exception:
                pass
                
            y, x = np.ogrid[:shape[0], :shape[1]]
            r2 = (x - center_x)**2 + (y - center_y)**2
            dispersion = 0.05
            wl_sigma = float(self.source_sigma())
            sigma = wl_sigma / dispersion
            
            # Spot peak intensity proportional to exposure time
            peak = 15000.0 * (self._exposure_time / 0.1)
            
            # Make the peak intensity vary with Top and Bottom gates to simulate quantum transport/gating
            try:
                smu = Instrument.find_instrument('smu')
                v_top_mv = smu.ch1.volt() * 1000.0
                v_bot_mv = smu.ch2.volt() * 1000.0
                # Create a diagonal charge transition line pattern (voltages are in mV now to match the sweep range)
                gate_factor = 0.1 + 0.9 * (np.exp(-((v_top_mv - v_bot_mv)**2) / 8.0) + 
                                           0.3 * np.exp(-((v_top_mv + v_bot_mv - 5.0)**2) / 5.0))
                #peak *= gate_factor
            except Exception:
                pass

                
            background = 200.0
            # Noise is disabled in simulation mode to ensure that the intensity remains mathematically
            # constant across sweeps and plots as a single uniform color.
            noise = 0.0
            
            frame = peak * np.exp(-r2 / (2 * sigma**2)) + background + noise
            # Clip pixel values between 0 and 65535 (16-bit CCD depth)
            return np.clip(frame, 0, 65535).astype(np.uint16)
        else:
            # Start acquisition and wait for frame
            self.cam.start_acquisition()
            self.cam.wait_for_frame()
            frame = self.cam.read_latest_image()
            self.cam.stop_acquisition()
            return frame
