from qcodes.instrument import VisaInstrument, Instrument

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
            self._positions = {1: 0, 2: 0, 3: 0}
        else:
            # Default Attocube serial settings: 9600 baud, CRLF terminator
            super().__init__(name, address, terminator='\r\n', **kwargs)
            
        # Add amplitude parameters for channels 1, 2, 3
        for axis in [1, 2, 3]:
            self.add_parameter(f'voltage_ch{axis}',
                               label=f'Axis {axis} Step Amplitude',
                               unit='V',
                               get_cmd=lambda a=axis: self._get_voltage(a),
                               set_cmd=lambda val, a=axis: self._set_voltage(a, val))
            
        if self.simulated:
            print(f"Initialized Attocube ANC150 '{name}' in Simulation Mode.")
        else:
            self.connect_message()

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
        # Parse return value, e.g., 'v = 30.0 V' or '30.0'
        parts = res.strip().split()
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
            self.write(f"setv {axis} {val:.1f}")

    def step(self, axis: int, steps: int) -> None:
        """
        Step the specified axis.
        Use positive integer for forward/up steps, and negative integer for backward/down.
        """
        if self.simulated:
            self._positions[axis] += steps
            print(f"Attocube ANC150 '{self.name}': Stepped Axis {axis} by {steps} steps (Simulated pos: {self._positions[axis]})")
        else:
            direction = "u" if steps > 0 else "d"
            self.write(f"step {axis} {direction} {abs(steps)}")
            
    def stop(self, axis: int) -> None:
        """Stop motion on the specified axis."""
        if self.simulated:
            print(f"Attocube ANC150 '{self.name}': Stopped Axis {axis} (Simulated)")
        else:
            self.write(f"stop {axis}")
