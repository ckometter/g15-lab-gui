try:
    import elliptec
except ImportError:
    elliptec = None

from qcodes.instrument import Instrument

class ThorlabsELL14(Instrument):
    """
    QCoDeS driver wrapper for the Thorlabs ELL14 Elliptec rotation stage.
    Requires the 'thorlabs-elliptec' library for physical connection.
    """
    def __init__(self, name, port=None, address='0', **kwargs):
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

        # Inherit from base Instrument since the serial port connection is 
        # managed by the underlying thorlabs-elliptec controller.
        super().__init__(name, **kwargs)
        
        self.simulated = (port is None or port == 'simulated')
        self.port = port
        self.dev_address = address
        
        if self.simulated:
            self._angle = 0.0
            print(f"Initialized Thorlabs ELL14 '{name}' in Simulation Mode.")
        else:
            if elliptec is None:
                raise ImportError(
                    "The 'thorlabs-elliptec' library is required to control physical ELL14 hardware. "
                    "Run 'pip install thorlabs-elliptec' to install it."
                )
            
            # Setup serial communication using the library
            self.controller = elliptec.Controller(port)
            self.device = elliptec.Rotator(self.controller, address=address)
            
            # Perform optional calibration/homing on start
            try:
                self.device.home()
            except Exception as e:
                print(f"Warning: Homing Thorlabs ELL14 '{name}' failed: {e}")
                
        self.add_parameter('angle',
                           label='Rotation Angle',
                           unit='deg',
                           get_cmd=self._get_angle,
                           set_cmd=self._set_angle)

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
            'vendor': 'Thorlabs',
            'model': 'ELL14',
            'serial': 'simulated' if self.simulated else self.dev_address,
            'firmware': None
        }


    def _get_angle(self) -> float:
        if self.simulated:
            return self._angle
        return float(self.device.get_angle())

    def _set_angle(self, val: float) -> None:
        if self.simulated:
            self._angle = val
        else:
            self.device.set_angle(val)
            
    def home(self) -> None:
        """Home the rotation stage."""
        if self.simulated:
            self._angle = 0.0
        else:
            self.device.home()
