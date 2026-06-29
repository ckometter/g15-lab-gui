from qcodes.instrument import VisaInstrument, Instrument
import re
import time

class AttocubeANC150(VisaInstrument):
    """
    QCoDeS driver for the Attocube ANC150 Piezo Positioner/Step Controller.
    Communicates via RS-232 serial.
    """
    def __init__(self, name, address, **kwargs):
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
            self._frequencies = {1: 150.0, 2: 150.0, 3: 150.0}
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
        resource, visabackend, resource_manager = super()._open_resource(address, visalib)
        if 'ASRL' in address:
            resource.baud_rate = 38400
        return resource, visabackend, resource_manager

    def ask_raw(self, cmd: str) -> str:
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

    def write_raw(self, cmd: str) -> None:
        self.ask_raw(cmd)

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

    def _get_voltage(self, axis: int) -> float:
        if self.simulated:
            return self._amplitudes[axis]
        
        res = self.ask(f"getv {axis}")
        lines = res.strip().split('\n')
        for line in lines:
            if ('voltage' in line or 'v' in line) and 'getv' not in line:
                parts = line.split()
                for p in parts:
                    try:
                        return float(p)
                    except ValueError:
                        pass
        raise ValueError(f"Could not parse voltage response: {res}")

    def _set_voltage(self, axis: int, val: float) -> None:
        if self.simulated:
            self._amplitudes[axis] = val
        else:
            self.write(f"setv {axis} {int(round(val))}")

    def step(self, axis: int, steps: int) -> None:
        """
        Step the specified axis.
        Use positive integer for forward/up steps, and negative integer for backward/down.
        """
        if self.simulated:
            self._positions[axis] += steps
            print(f"Attocube ANC150 '{self.name}': Stepped Axis {axis} by {steps} steps (Simulated pos: {self._positions[axis]})")
        else:
            self.write(f"setm {axis} stp")
            cmd = "stepu" if steps > 0 else "stepd"
            self.write(f"{cmd} {axis} {int(abs(steps))}")

    def _get_regulator_offset(self) -> float:
        # Front-panel frequency regulator offset (not supported on this firmware)
        return 0.0

    def _get_frequency(self, axis: int) -> float:
        """Return stepping frequency (controller)."""
        if self.simulated:
            return self._frequencies[axis]
        try:
            res = self.ask(f"getf {axis}")
            lines = res.strip().split('\n')
            for line in lines:
                if ('frequency' in line or 'f' in line) and 'getf' not in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            return float(p)
                        except ValueError:
                            pass
            raise ValueError(f"Could not parse frequency response: {res}")
        except Exception as e:
            print(f"Error getting frequency for axis {axis}: {e}")
            return 0.0

    def _set_frequency(self, axis: int, val: float) -> None:
        """Set stepping frequency."""
        if self.simulated:
            self._frequencies[axis] = val
        else:
            self.write(f'setf {axis} {val:.0f}')

    def start_continuous(self, axis: int, direction: int) -> None:
        """
        Start continuous stepping on the specified axis.
        Use direction > 0 for forward/up, direction < 0 for backward/down.
        """
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
