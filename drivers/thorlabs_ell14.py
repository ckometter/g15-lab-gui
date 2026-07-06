import time
import logging
import threading

try:
    import elliptec
except ImportError:
    elliptec = None

from qcodes.instrument import Instrument
from qcodes import validators as vals

log = logging.getLogger(__name__)

class ThorlabsELL14(Instrument):
    """
    QCoDeS driver wrapper for the Thorlabs ELL14 Elliptec rotation stage.
    Hardened against initialization snapshots and transient hardware drops.
    """
    _controllers = {}
    _locks = {}
    def __init__(self, name, port=None, address='0', **kwargs):
        try:
            instr = Instrument.find_instrument(name)
            try: Instrument.remove_instance(instr)
            except Exception: pass
            try: instr.close()
            except Exception: pass
        except KeyError:
            pass

        super().__init__(name, **kwargs)
        
        self.simulated = (port is None or port == 'simulated')
        self.port = port
        self.dev_address = address
        
        if self.simulated:
            self._angle = 0.0
            print(f"Initialized Thorlabs ELL14 '{name}' in Simulation Mode.")
        else:
            if elliptec is None:
                raise ImportError("The 'thorlabs-elliptec' library is required for physical hardware.")
            
            print(f"Connecting to Thorlabs ELL14 '{name}' on {port}...")
            if port not in ThorlabsELL14._controllers:
                ThorlabsELL14._controllers[port] = elliptec.Controller(port)
                ThorlabsELL14._locks[port] = threading.Lock()
            self.controller = ThorlabsELL14._controllers[port]
            self.lock = ThorlabsELL14._locks[port]
            self.device = elliptec.Rotator(self.controller, address=address)
            time.sleep(1.0) # Settle time
                
        self.add_parameter('angle',
                           label='Rotation Angle',
                           unit='deg',
                           get_cmd=self._get_angle,
                           set_cmd=self._set_angle,
                           vals=vals.Numbers(-180.0, 180.0))

    def close(self) -> None:
        if getattr(self, 'simulated', False):
            Instrument.close(self)
        else:
            try: super().close()
            except Exception: self.remove_instance(self)

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
        
        # DEFENSIVE FIX: Protect against QCoDeS Snapshot mechanical crash
        try:
            with self.lock:
                return float(self.device.get_angle())
        except Exception as e:
            print(f"\n[Hardware Warning] '{self.name}' position read failed: {e}")
            print("The stage may be mechanically locked, un-homed, or power-cycled.")
            return 0.0 # Return fallback value so station_setup survives

    def _set_angle(self, val: float) -> None:
        if self.simulated:
            self._angle = val
        else:
            try:
                with self.lock:
                    self.device.set_angle(val)
                self.wait_until_idle()
            except Exception as e:
                print(f"ERROR: Failed to move '{self.name}' to {val}: {e}")
            
    def wait_until_idle(self, timeout: float = 15.0) -> None:
        if self.simulated:
            return
            
        start_time = time.time()
        last_pos = self._get_angle()
        time.sleep(0.1) 
        
        while time.time() - start_time < timeout:
            current_pos = self._get_angle()
            if abs(current_pos - last_pos) < 0.01:
                return
            last_pos = current_pos
            time.sleep(0.05)
            
        raise TimeoutError(f"Stage failed to stabilize position within {timeout} seconds.")

    def home(self) -> None:
        """Home the rotation stage and block until it arrives at 0.0."""
        if self.simulated:
            self._angle = 0.0
        else:
            print(f"Homing physical stage '{self.name}'...")
            try:
                with self.lock:
                    self.device.home()
                self.wait_until_idle(timeout=15.0)
                print("Homing sequence complete.")
            except Exception as e:
                print(f"Homing Failed: {e}. Check power and mechanical obstructions.")