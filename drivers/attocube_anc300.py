import threading
import math
from qcodes.instrument import Instrument
from scanner_safety import SafeScannerMotion

try:
    from qcodes_contrib_drivers.drivers.Attocube.ANC300 import ANC300 as PhysicalANC300
except ImportError:
    PhysicalANC300 = None

if PhysicalANC300 is not None:
    class AttocubeANC300(SafeScannerMotion, PhysicalANC300):
        """
        QCoDeS driver for the Attocube ANC300 Piezo Controller.
        Integrates physical driver (from qcodes_contrib_drivers) and simulation fallback.
        """
        def __init__(self, name, address, **kwargs):
            self.comm_lock = threading.RLock()
            self.init_motion_safety()
            self._voltages = {1: 0.0, 2: 0.0, 3: 0.0} if address == "simulated" else {}
            # Close any existing instrument with the same name to avoid collision
            try:
                instr = Instrument.find_instrument(name)
                instr.close()
            except KeyError:
                pass
            except Exception:
                try:
                    Instrument.remove_instance(instr)
                except Exception:
                    pass

            self.simulated = (address == 'simulated')
            if self.simulated:
                # Bypass physical VisaInstrument initialization
                Instrument.__init__(self, name, **kwargs)
                for axis in [1, 2, 3]:
                    self.add_parameter(f'voltage_ch{axis}',
                                       label=f'Axis {axis} Sweep Voltage',
                                       unit='V',
                                       get_cmd=lambda a=axis: self.get_voltage(a),
                                       set_cmd=lambda val, a=axis: self.set_voltage(a, val))
                print(f"Initialized Attocube ANC300 '{name}' in Simulation Mode.")
            else:
                super().__init__(name, address, **kwargs)
                # Disable snapshot queries for trigger parameters to avoid hardware warning errors
                for ax_name in ['axis1', 'axis2', 'axis3']:
                    ax = getattr(self, ax_name, None)
                    if not ax and ax_name in self.submodules:
                        ax = self.submodules[ax_name]
                    if ax and hasattr(ax, 'parameters'):
                        for p_name in ['triggerUp', 'triggerDown']:
                            if p_name in ax.parameters:
                                ax.parameters[p_name].snapshot_get = False
                                ax.parameters[p_name]._snapshot_get = False
                for submod in self.submodules.values():
                    if hasattr(submod, 'parameters'):
                        for p_name in ['triggerUp', 'triggerDown']:
                            if p_name in submod.parameters:
                                submod.parameters[p_name].snapshot_get = False
                                submod.parameters[p_name]._snapshot_get = False

        def _open_resource(self, address: str, visalib: str | None):
            opened = super()._open_resource(address, visalib)
            # qcodes 0.51 returns (resource, backend, manager); qcodes 0.59 returns the resource.
            resource = opened[0] if isinstance(opened, tuple) else opened
            if 'ASRL' in address:
                resource.baud_rate = 38400
            return opened

        def _clear_input_buffer(self):
            if hasattr(self, 'visa_handle') and self.visa_handle is not None:
                try:
                    bytes_to_read = getattr(self.visa_handle, 'bytes_in_buffer', 0)
                    if bytes_to_read > 0:
                        self.visa_handle.read_bytes(bytes_to_read)
                except Exception:
                    pass

        def _exchange(self, cmd):
            """Consume a complete reply. Never retry a movement/write command."""
            with self.comm_lock:
                self._clear_input_buffer()
                self.visa_handle.write(cmd)
                payload = []
                for _ in range(32):
                    # A timeout is a failure, not an acknowledgement.
                    line = self.visa_handle.read().strip().lstrip('> ').strip()
                    if not line or line == cmd:
                        continue
                    if line == 'OK':
                        response = ' '.join(payload)
                        if '=' in response:
                            return response.split('=', 1)[1].strip().split()[0]
                        return response
                    if line.startswith('ERROR'):
                        raise RuntimeError(f"ANC300 rejected {cmd!r}: {' '.join(payload)} {line}")
                    payload.append(line)
                raise RuntimeError(f"ANC300 did not acknowledge {cmd!r}.")

        def ask_raw(self, cmd: str) -> str:
            return self._exchange(cmd)

        def write_raw(self, cmd: str) -> None:
            parts = cmd.split()
            if parts and parts[0].lower() == 'seta':
                if len(parts) != 3:
                    raise ValueError('Expected seta axis voltage.')
                self.set_voltage(int(parts[1]), float(parts[2]))
                return
            self._exchange(cmd)

        @staticmethod
        def _validate_voltage(axis, value):
            if not isinstance(axis, int) or axis not in range(1, 8):
                raise ValueError("ANC300 axis must be an integer from 1 to 7.")
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 150:
                raise ValueError("ANC300 voltage must be finite and within 0 to 150 V.")
            return value

        def get_voltage(self, axis: int) -> float:
            """Read the configured DC offset. Never substitute cached/zero values."""
            self._validate_voltage(axis, 0)
            with self.comm_lock:
                if self.simulated:
                    if axis not in self._voltages:
                        raise ValueError(f"No simulated axis {axis}.")
                    return self._voltages[axis]
                value = float(self.ask(f'geta {axis}'))
                self._validate_voltage(axis, value)
                self._voltages[axis] = value
                return value

        def get_axis_status(self, axis):
            with self.comm_lock:
                offset = self.get_voltage(axis)
                if self.simulated:
                    return {'offset': offset, 'output': offset, 'mode': 'off'}
                output = float(self.ask(f'geto {axis}'))
                if not math.isfinite(output):
                    raise RuntimeError(f"Invalid ANC300 axis {axis} output readback.")
                mode = str(self.ask(f'getm {axis}')).strip()
                return {'offset': offset, 'output': output, 'mode': mode}

        def _write_voltage_step(self, axis: int, val: float) -> None:
            """Set a DC offset only in offset mode and verify its readback."""
            value = self._validate_voltage(axis, val)
            with self.comm_lock:
                state = self.get_axis_status(axis)
                if state['mode'] != 'off':
                    raise RuntimeError(f"ANC300 axis {axis} is in {state['mode']!r} mode, not offset mode ('off'). No voltage or mode was changed.")
                if self.simulated:
                    self._voltages[axis] = value
                    return
                self._exchange(f'seta {axis} {value:.6f}')
                actual = self.get_voltage(axis)
                if not math.isclose(actual, value, rel_tol=0, abs_tol=0.01):
                    raise RuntimeError(f"ANC300 axis {axis} offset readback is {actual:.3f} V after requesting {value:.3f} V. Check the controller before retrying.")

        def _set_voltage(self, val: float) -> None:
            # Backward compatibility method (acts on channel 1)
            self.set_voltage(1, val)

        def reset_communication(self) -> bool:
            """Reset and resynchronize serial communication with the ANC300 controller."""
            if getattr(self, 'simulated', False) or not hasattr(self, 'comm_lock'):
                return True
            with self.comm_lock:
                try:
                    # 1. Clear input buffer
                    self._clear_input_buffer()

                    # 2. Clear PyVISA handle session if available
                    if hasattr(self, 'visa_handle') and self.visa_handle is not None:
                        try:
                            self.visa_handle.clear()
                        except Exception:
                            pass

                    # 3. Probe query to resynchronize command/response prompt stream
                    self.ask_raw('geta 1')
                    self._clear_input_buffer()
                    return True
                except Exception as e:
                    print(f"AttocubeANC300 reset_communication error: {e}")
                    return False

        def close(self) -> None:
            if getattr(self, 'simulated', False):
                Instrument.close(self)
            else:
                try:
                    super().close()
                except Exception:
                    self.remove_instance(self)

        def get_idn(self) -> dict:
            if getattr(self, 'simulated', False):
                return {
                    'vendor': 'Attocube',
                    'model': 'ANC300',
                    'serial': 'simulated',
                    'firmware': None
                }
            try:
                return super().get_idn()
            except Exception:
                return {
                    'vendor': 'Attocube',
                    'model': 'ANC300',
                    'serial': 'unknown',
                    'firmware': None
                }
else:
    class AttocubeANC300(SafeScannerMotion, Instrument):
        """
        Simulated mockup driver fallback for the Attocube ANC300 Piezo Controller.
        """
        def __init__(self, name, address=None, **kwargs):
            # Close any existing instrument with the same name to avoid collision
            try:
                instr = Instrument.find_instrument(name)
                instr.close()
            except KeyError:
                pass
            except Exception:
                try:
                    Instrument.remove_instance(instr)
                except Exception:
                    pass

            super().__init__(name, **kwargs)
            self.comm_lock = threading.RLock()
            self.init_motion_safety()
            self.simulated = True
            self._voltages = {1: 0.0, 2: 0.0, 3: 0.0}
            for axis in [1, 2, 3]:
                self.add_parameter(f'voltage_ch{axis}',
                                   label=f'Axis {axis} Sweep Voltage',
                                   unit='V',
                                   get_cmd=lambda a=axis: self.get_voltage(a),
                                   set_cmd=lambda val, a=axis: self.set_voltage(a, val))
            print(f"Initialized Attocube ANC300 '{name}' in Simulation Mode (fallback).")

        def get_voltage(self, axis: int) -> float:
            """Get fine scanning DC offset voltage for the given axis (1, 2, or 3)."""
            return self._voltages[axis]

        def _write_voltage_step(self, axis: int, val: float) -> None:
            """Set fine scanning DC offset voltage for the given axis (1, 2, or 3)."""
            self._voltages[axis] = val

        def get_axis_status(self, axis):
            offset = self.get_voltage(axis)
            return {'offset': offset, 'output': offset, 'mode': 'off'}

        def _set_voltage(self, val: float) -> None: #internal helper function
            # Backward compatibility method (acts on channel 1)
            self.set_voltage(1, val)

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
                'model': 'ANC300',
                'serial': 'simulated',
                'firmware': None
            }

