from qcodes.instrument import VisaInstrument, Instrument
import re
import math
import time
import threading

MAX_FREQUENCY = 100
MAX_AMPLITUDE = 120

class AttocubeANC150(VisaInstrument):
    """
    QCoDeS driver for the Attocube ANC150 Piezo Positioner/Step Controller.
    Communicates via RS-232 serial.
    """
    def __init__(self, name, address, **kwargs):
        self.comm_lock = threading.RLock()
        self.motion_lock = threading.Lock()
        self._pseudo_positions = {1: 0, 2: 0, 3: 0}
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

        self.simulated = (address == 'simulated')
        
        if self.simulated:
            Instrument.__init__(self, name, **kwargs)
            self._amplitudes = {1: 30.0, 2: 30.0, 3: 30.0}
            self._frequencies = {1: 100.0, 2: 100.0, 3: 100.0}
            self._positions = {1: 0, 2: 0, 3: 0}
        else:
            # Default Attocube serial settings: 38400 baud, CR terminator
            super().__init__(name, address, terminator='\r', **kwargs)
            try:
                time.sleep(0.2)
                self.visa_handle.flush(2)  # VI_READ_BUF_DISCARD = 2
            except Exception:
                pass
            
        # Add amplitude parameters for channels 1, 2, 3
        for axis in [1, 2, 3]:
            self.add_parameter(f'voltage_ch{axis}',
                               label=f'Axis {axis} Step Amplitude',
                               unit='V',
                               get_cmd=lambda a=axis: self._get_voltage(a),
                               set_cmd=lambda val, a=axis: self._set_voltage(a, val),
                               snapshot_get=False)
            
        # Add frequency parameters for channels 1, 2, 3
        for axis in [1, 2, 3]:
            self.add_parameter(f'frequency_ch{axis}',
                               label=f'Axis {axis} Step Frequency',
                               unit='Hz',
                               get_cmd=lambda a=axis: self._get_frequency(a),
                               set_cmd=lambda val, a=axis: self._set_frequency(a, val),
                               snapshot_get=False)
            
        if self.simulated:
            print(f"Initialized Attocube ANC150 '{name}' in Simulation Mode.")
        else:
            self.connect_message()

    def _open_resource(self, address: str, visalib: str | None):
        resource = super()._open_resource(address, visalib)
        if 'ASRL' in address:
            resource.baud_rate = 38400
        return resource

    def ask_raw(self, cmd: str) -> str:
        lock = getattr(self, 'comm_lock', None)
        if lock is not None:
            lock.acquire()
        try:
            self.visa_handle.write(cmd)
            res = ""
            while True:
                char = self.visa_handle.read_bytes(1).decode('ascii', errors='ignore')
                if not char:
                    break
                res += char
                if res.endswith("> "):
                    break
            return res
        finally:
            if lock is not None:
                lock.release()

    def write_raw(self, cmd: str) -> None:
        response = self.ask_raw(cmd)
        if 'error' in response.lower() or 'OK' not in response:
            raise RuntimeError(f"ANC150 rejected {cmd!r}: {response.strip()}")

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
            'vendor': 'Attocube',
            'model': 'ANC150',
            'serial': 'simulated' if self.simulated else 'unknown',
            'firmware': None
        }

    def snapshot_base(self, update = False, params_to_skip_update = None) -> dict:
        if self.simulated:
            return Instrument.snapshot_base(self, update=update, params_to_skip_update=params_to_skip_update)
        return super().snapshot_base(update=update, params_to_skip_update=params_to_skip_update)

    def _read_numeric(self, command, field):
        response = self.ask(command)
        if 'error' in response.lower() or 'OK' not in response:
            raise RuntimeError(f"ANC150 readback failed: {response.strip()}")
        match = re.search(rf'{field}\s*=\s*([-+0-9.eE]+)', response, re.I)
        if match is None or not math.isfinite(float(match[1])):
            raise RuntimeError(f"Invalid ANC150 {field} readback: {response.strip()}")
        return float(match[1])

    def _get_voltage(self, axis: int) -> float:
        if self.simulated:
            return self._amplitudes[axis]
        return self._read_numeric(f'getv {axis}', 'voltage')

    def _set_voltage(self, axis: int, val: float) -> None:
        val = self._validate_setting(axis, val, 0, MAX_AMPLITUDE, 'amplitude (V)')
        if self.motion_lock.locked():
            raise RuntimeError('Stop stepper motion before changing amplitude.')
        if self.simulated:
            self._amplitudes[axis] = val
        else:
            self.write(f"setv {axis} {int(round(val))}")

    def step(self, axis: int, steps: int) -> None:
        """
        Step the specified axis.
        Use positive integer for forward/up steps, and negative integer for backward/down.
        """
        self._validate_axis(axis)
        if type(steps) is not int:
            raise ValueError('Stepper motion requires an integer step count.')
        if steps == 0:
            return
        with self.comm_lock:
            self.validate_drive(axis)
            try:
                if self.simulated:
                    self._positions[axis] += steps
                else:
                    self.write(f"setm {axis} stp")
                    cmd = "stepu" if steps > 0 else "stepd"
                    self.write(f"{cmd} {axis} {abs(steps)}")
            except Exception:
                self._pseudo_positions[axis] = None
                raise
            if self._pseudo_positions[axis] is not None:
                self._pseudo_positions[axis] += steps

    @staticmethod
    def _validate_axis(axis):
        if type(axis) is not int or axis not in (1, 2, 3):
            raise ValueError('Stepper axis must be 1, 2 or 3.')

    @classmethod
    def _validate_setting(cls, axis, value, minimum, maximum, label):
        cls._validate_axis(axis)
        value = float(value)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f'Stepper {label} must be {minimum} to {maximum}.')
        return value

    def validate_drive(self, axis):
        frequency = self._validate_setting(axis, self._get_frequency(axis), 1, MAX_FREQUENCY, 'frequency (Hz)')
        self._validate_setting(axis, self._get_voltage(axis), 0, MAX_AMPLITUDE, 'amplitude (V)')
        return frequency

    def pseudo_position(self, axis):
        self._validate_axis(axis)
        return self._pseudo_positions[axis]

    def zero_pseudo_position(self):
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError('Stop stepper motion before zeroing.')
        try:
            with self.comm_lock:
                self._pseudo_positions = {1: 0, 2: 0, 3: 0}
        finally:
            self.motion_lock.release()

    def _get_regulator_offset(self) -> float:
        # Front-panel frequency regulator offset (not supported on this firmware)
        return 0.0

    def _get_frequency(self, axis: int) -> float:
        if self.simulated:
            return self._frequencies[axis]
        return self._read_numeric(f'getf {axis}', 'frequency')

    def _set_frequency(self, axis: int, val: float) -> None:
        """Set stepping frequency."""
        val = self._validate_setting(axis, val, 1, MAX_FREQUENCY, 'frequency (Hz)')
        if self.motion_lock.locked():
            raise RuntimeError('Stop stepper motion before changing frequency.')
        if self.simulated:
            self._frequencies[axis] = val
        else:
            self.write(f'setf {axis} {val:.0f}')

    def start_continuous(self, axis: int, direction: int) -> None:
        """
        Start continuous stepping on the specified axis.
        Use direction > 0 for forward/up, direction < 0 for backward/down.
        """
        self.validate_drive(axis)
        if direction not in (-1, 1):
            raise ValueError('Stepper direction must be -1 or +1.')
        # Uncounted continuous motion cannot retain a meaningful coordinate.
        self._pseudo_positions[axis] = None
        if self.simulated:
            print(f"Attocube ANC150 '{self.name}': Started continuous stepping on Axis {axis} (Simulated)")
        else:
            self.write(f"setm {axis} stp")
            cmd = "stepu" if direction > 0 else "stepd"
            self.write(f"{cmd} {axis} c")

    def stop(self, axis: int) -> None:
        """Stop motion on the specified axis."""
        if self.simulated:
            print(f"Attocube ANC150 '{self.name}': Stopped Axis {axis} (Simulated)")
        else:
            self.write(f"stop {axis}")
