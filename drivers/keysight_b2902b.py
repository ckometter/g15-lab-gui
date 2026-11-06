import math
import threading
from qcodes.instrument import VisaInstrument, Instrument
from qcodes.instrument import InstrumentChannel
from qcodes.validators import Numbers

from gate_safety import (
    DEFAULT_GATE_RAMP_RATE_V_PER_S,
    GATE_CURRENT_COMPLIANCE_A,
    GATE_MIN_CURRENT_COMPLIANCE_A,
    GATE_MAX_CURRENT_COMPLIANCE_A,
    GATE_MAX_ABS_VOLTAGE_V,
    GATE_MAX_RAMP_STEP_V,
    GateRampCancelled,
    validate_gate_voltage,
    validate_positive_ramp_value,
    validate_current_compliance,
)


class RampedSet:
    """Compatibility API: direct voltage writes use this same safe ramp."""
    def __init__(self, channel):
        self.channel = channel

    def volt(self, target_voltage: float, callback=None):
        self.channel._ramp_parameter('volt', target_voltage, callback)

    def curr(self, target_current: float, callback=None):
        self.channel._ramp_parameter('curr', target_current, callback)


class B2902BChannel(InstrumentChannel):
    """
    Represent a single channel of the Keysight B2902B Low Noise Power Source.
    """
    def __init__(self, parent, name, channel_num):
        super().__init__(parent, name)
        self.channel_num = channel_num
        self.simulated = parent.simulated
        self.ramping = False
        self.ramp_stop_requested = False
        self._enable_in_progress = False
        self._ramp_lock = threading.RLock()
        self.current_compliance_target = GATE_CURRENT_COMPLIANCE_A
        self.set_r = RampedSet(self)
        
        if self.simulated:
            self._volt = 0.0
            self._curr = 0.0
            self._output = 0
            self._curr_limit = GATE_CURRENT_COMPLIANCE_A
            self._volt_limit = 10.0
            self._source_mode = 'VOLT'
            
            self.add_parameter('volt',
                               label=f'Channel {channel_num} Voltage',
                               unit='V',
                               get_cmd=lambda: self._volt,
                               set_cmd=self._set_volt,
                               vals=Numbers(-GATE_MAX_ABS_VOLTAGE_V, GATE_MAX_ABS_VOLTAGE_V))
            
            self.add_parameter('curr',
                               label=f'Channel {channel_num} Current',
                               unit='A',
                               get_cmd=lambda: self._curr,
                               set_cmd=self._set_curr,
                               vals=Numbers(-GATE_CURRENT_COMPLIANCE_A, GATE_CURRENT_COMPLIANCE_A))
            
            self.add_parameter('source_mode',
                               label=f'Channel {channel_num} Source Mode',
                               get_cmd=lambda: self._source_mode,
                               set_cmd=self._set_source_mode,
                               val_mapping={'VOLT': 'VOLT', 'CURR': 'CURR'})

            self.add_parameter('output',
                               label=f'Channel {channel_num} Output State',
                               get_cmd=lambda: self._output,
                               set_cmd=self._set_output)
            
            self.add_parameter('curr_limit',
                               label=f'Channel {channel_num} Current Limit',
                               unit='A',
                               get_cmd=lambda: self._curr_limit,
                               set_cmd=self._set_curr_limit,
                               vals=Numbers(GATE_MIN_CURRENT_COMPLIANCE_A, GATE_MAX_CURRENT_COMPLIANCE_A))
            
            self.add_parameter('volt_limit',
                               label=f'Channel {channel_num} Voltage Limit',
                               unit='V',
                               get_cmd=lambda: self._volt_limit,
                               set_cmd=self._set_volt_limit,
                               vals=Numbers(0, GATE_MAX_ABS_VOLTAGE_V))
            
            self.add_parameter('measure_volt',
                               label=f'Channel {channel_num} Measured Voltage',
                               unit='V',
                               get_cmd=self._measure_volt)
            
            self.add_parameter('measure_curr',
                               label=f'Channel {channel_num} Measured Current',
                               unit='A',
                               get_cmd=self._measure_curr)
        else:
            # Sourcing mode selection
            self.add_parameter('source_mode',
                               label=f'Channel {channel_num} Source Mode',
                               get_cmd=f':SOUR{channel_num}:FUNC:MODE?',
                               set_cmd=self._set_source_mode,
                               val_mapping={'VOLT': 'VOLT', 'CURR': 'CURR'})

            # Active voltage sourcing commands
            self.add_parameter('volt',
                               label=f'Channel {channel_num} Voltage',
                               unit='V',
                               get_cmd=f':SOUR{channel_num}:VOLT?',
                               set_cmd=self._set_volt,
                               get_parser=float,
                               vals=Numbers(-GATE_MAX_ABS_VOLTAGE_V, GATE_MAX_ABS_VOLTAGE_V))
            
            # Active current sourcing commands
            self.add_parameter('curr',
                               label=f'Channel {channel_num} Current',
                               unit='A',
                               get_cmd=f':SOUR{channel_num}:CURR?',
                               set_cmd=self._set_curr,
                               get_parser=float,
                               vals=Numbers(-GATE_CURRENT_COMPLIANCE_A, GATE_CURRENT_COMPLIANCE_A))
            
            self.add_parameter('output',
                               label=f'Channel {channel_num} Output State',
                               get_cmd=f':OUTP{channel_num}?',
                               set_cmd=self._set_output,
                               val_mapping={True: 1, False: 0})
            
            # Current compliance limit (used in voltage sourcing mode)
            self.add_parameter('curr_limit',
                               label=f'Channel {channel_num} Current Limit',
                               unit='A',
                               get_cmd=f':SENS{channel_num}:CURR:PROT?',
                               set_cmd=self._set_curr_limit,
                               get_parser=float,
                               vals=Numbers(GATE_MIN_CURRENT_COMPLIANCE_A, GATE_MAX_CURRENT_COMPLIANCE_A))

            # Voltage compliance limit (used in current sourcing mode)
            self.add_parameter('volt_limit',
                               label=f'Channel {channel_num} Voltage Limit',
                               unit='V',
                               get_cmd=f':SENS{channel_num}:VOLT:PROT?',
                               set_cmd=self._set_volt_limit,
                               get_parser=float,
                               vals=Numbers(0, GATE_MAX_ABS_VOLTAGE_V))
            
            # Active measurement commands
            self.add_parameter('measure_volt',
                               label=f'Channel {channel_num} Measured Voltage',
                               unit='V',
                               get_cmd=f':MEAS:VOLT? (@{channel_num})',
                               get_parser=float)
            
            self.add_parameter('measure_curr',
                               label=f'Channel {channel_num} Measured Current',
                               unit='A',
                               get_cmd=f':MEAS:CURR? (@{channel_num})',
                               get_parser=float)

        # Ramping parameters
        self._ramp_rate = DEFAULT_GATE_RAMP_RATE_V_PER_S
        self._ramp_step = GATE_MAX_RAMP_STEP_V
        
        self.add_parameter('ramp_rate',
                           label=f'Channel {channel_num} Ramp Rate',
                           unit='V/s or A/s',
                           get_cmd=self._get_ramp_rate,
                           set_cmd=self._set_ramp_rate)
        
        self.add_parameter('ramp_step',
                           label=f'Channel {channel_num} Ramp Step',
                           unit='V or A',
                           get_cmd=self._get_ramp_step,
                           set_cmd=self._set_ramp_step)
                           
        # Load initial values from value_storage if available
        self.ramp_rate()
        self.ramp_step()

    def _set_volt(self, val):
        self._ramp_parameter('volt', val)

    def _check_ramp_cancelled(self):
        if self.ramp_stop_requested:
            raise GateRampCancelled(f'Channel {self.channel_num}: ramp stopped at its last setpoint.')

    def _wait_for_ramp(self, delay):
        import time
        # Small sleeps allow stop/pause to interrupt even a very slow ramp.
        remaining = delay
        while remaining > 0:
            self._check_ramp_cancelled()
            interval = min(remaining, 0.05)
            time.sleep(interval)
            remaining -= interval
        self._check_ramp_cancelled()

    def _ramp_parameter(self, parameter_name, target, callback=None):
        # All validation that does not need a readback precedes every command.
        target = float(target)
        if parameter_name == 'volt':
            validate_gate_voltage(target, self.channel_num)
        elif not math.isfinite(target) or abs(target) > GATE_CURRENT_COMPLIANCE_A:
            raise ValueError('Gate source current exceeds the code-only current limit.')
        ramp_rate = validate_positive_ramp_value(self.ramp_rate(), 'Ramp rate')
        configured_step = validate_positive_ramp_value(self.ramp_step(), 'Ramp step')
        maximum_step = min(configured_step, GATE_MAX_RAMP_STEP_V if parameter_name == 'volt'
                           else GATE_CURRENT_COMPLIANCE_A / 10)
        with self._ramp_lock:
            self._check_ramp_cancelled()
            self.ramping = True
            try:
                param = getattr(self, parameter_name)
                # Explicit hardware query: QCoDeS cache may describe an old session.
                initial = float(param())
                if parameter_name == 'volt':
                    validate_gate_voltage(initial, self.channel_num)
                elif not math.isfinite(initial) or abs(initial) > GATE_CURRENT_COMPLIANCE_A:
                    raise RuntimeError('Actual source current is outside the code-only limit.')
                self.enforce_current_compliance()
                if parameter_name == 'curr':
                    self.enforce_voltage_compliance()
                distance = target - initial
                count = math.ceil(abs(distance) / maximum_step)
                previous = initial
                for index in range(1, count + 1):
                    self._check_ramp_cancelled()
                    value = target if index == count else initial + distance * index / count
                    # Wait BEFORE each change, including small one-increment ramps.
                    if not self.simulated:
                        self._wait_for_ramp(abs(value - previous) / ramp_rate)
                    self._check_ramp_cancelled()
                    with self.parent._lock:
                        self._check_ramp_cancelled()
                        if parameter_name == 'volt':
                            self._write_voltage_step(value)
                        else:
                            self._set_curr(value)
                    param.cache.set(value)
                    previous = value
                    if callback is not None:
                        try:
                            callback(value)
                        except Exception:
                            pass
                if not count and callback is not None:
                    callback(initial)
                if count:
                    self._check_ramp_cancelled()
                    actual = float(param())
                    if (not math.isfinite(actual) or
                            not math.isclose(actual, target, rel_tol=1e-7, abs_tol=1e-9)):
                        raise RuntimeError(f'Channel {self.channel_num}: final {parameter_name} '
                                           f'setpoint not confirmed (requested {target:g}, '
                                           f'read {actual:g}). Operation blocked.')
            finally:
                self.ramping = False

    def _write_voltage_step(self, value):
        # This private primitive is only used after the ramp's compliance check.
        validate_gate_voltage(value, self.channel_num)
        if self.simulated:
            self._volt = value
        else:
            self.write(f':SOUR{self.channel_num}:VOLT {value:.12e}')

    def _set_curr(self, val):
        if not math.isfinite(val) or abs(val) > GATE_CURRENT_COMPLIANCE_A:
            raise ValueError('Gate source current exceeds the code-only current limit.')
        with self.parent._lock:
            self.enforce_voltage_compliance()
            if self.simulated:
                self._curr = val
            else:
                self.write(f':SOUR{self.channel_num}:CURR {val:.12e}')

    def _set_source_mode(self, val):
        with self.parent._lock:
            if val == 'CURR':
                self._check_source_current()
                self.enforce_voltage_compliance()
            else:
                validate_gate_voltage(self.volt(), self.channel_num)
                self.enforce_current_compliance()
            if self.source_mode() != val and self.output():
                raise RuntimeError('Turn the channel output off before changing source mode; '
                                   'a mode switch with output enabled can cause a voltage jump.')
            if self.simulated:
                self._source_mode = val
            else:
                self.write(f':SOUR{self.channel_num}:FUNC:MODE {val}')

    def _set_output(self, val):
        if not val:
            # Emergency OFF must remain immediate and also cancel a pending ON.
            with self.parent._lock:
                if self.ramping or self._enable_in_progress:
                    self.ramp_stop_requested = True
                self._write_output_state(False)
            return
        with self._ramp_lock:
            self._enable_in_progress = True
            try:
                self._check_ramp_cancelled()
                target = validate_gate_voltage(self.volt(), self.channel_num)
                self.enforce_current_compliance()
                source_mode = self.source_mode()
                if source_mode == 'CURR':
                    self._check_source_current()
                    self.enforce_voltage_compliance()
                if self.output():
                    return
                # Enabling a nonzero dormant voltage would bypass the ramp by
                # connecting the sample directly to that setpoint. Start at 0 V.
                if source_mode == 'VOLT':
                    self._ramp_parameter('volt', 0.0)
                with self.parent._lock:
                    self._check_ramp_cancelled()
                    self._write_output_state(True)
                if source_mode == 'VOLT':
                    self._ramp_parameter('volt', target)
            finally:
                self._enable_in_progress = False

    def _write_output_state(self, val):
        if self.simulated:
            self._output = int(val)
        else:
            self.write(f':OUTP{self.channel_num} {int(val)}')

    def _set_curr_limit(self, val):
        val = validate_current_compliance(val)
        with self.parent._lock:
            if self.simulated:
                self._curr_limit = val
            else:
                self.write(f':SENS{self.channel_num}:CURR:PROT {val:.6e}')
            actual = float(self.curr_limit())
            if not math.isfinite(actual) or not math.isclose(actual, val, rel_tol=1e-6, abs_tol=0):
                raise RuntimeError(f'Channel {self.channel_num}: {val:g} A compliance not confirmed '
                                   f'(read back {actual:g} A). Operation blocked.')
            # Change the runtime target only after the instrument confirms it.
            self.current_compliance_target = val

    def enforce_current_compliance(self):
        """Verify the selected runtime limit before ramps and output changes."""
        with self.parent._lock:
            self.curr_limit(self.current_compliance_target)

    def _check_source_current(self):
        actual = float(self.curr())
        if not math.isfinite(actual) or abs(actual) > GATE_CURRENT_COMPLIANCE_A:
            raise RuntimeError(f'Channel {self.channel_num}: source current exceeds {GATE_CURRENT_COMPLIANCE_A:g} A. '
                               'Reduce the current setpoint before enabling current sourcing.')

    def _set_volt_limit(self, val):
        val = validate_gate_voltage(val, self.channel_num)
        if val <= 0:
            raise ValueError('Voltage compliance must be greater than zero.')
        if self.simulated:
            self._volt_limit = val
        else:
            self.write(f':SENS{self.channel_num}:VOLT:PROT {val:.12e}')

    def enforce_voltage_compliance(self):
        """Current sourcing must not bypass the physical gate-voltage cap."""
        with self.parent._lock:
            actual = float(self.volt_limit())
            if math.isfinite(actual) and 0 < actual <= GATE_MAX_ABS_VOLTAGE_V:
                return
            self.volt_limit(GATE_MAX_ABS_VOLTAGE_V)
            actual = float(self.volt_limit())
            if not math.isfinite(actual) or not 0 < actual <= GATE_MAX_ABS_VOLTAGE_V:
                raise RuntimeError(f'Channel {self.channel_num}: voltage compliance not '
                                   f'confirmed below {GATE_MAX_ABS_VOLTAGE_V:g} V. Operation blocked.')

    def _get_ramp_rate(self):
        import os
        import json
        try:
            driver_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_dir = os.path.dirname(driver_dir)
            storage_path = os.path.join(workspace_dir, "value_storage")
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    data = json.load(f)
                    dg_data = data.get("dual_gate", {})
                    val = dg_data.get(f"ramprate{self.channel_num}")
                    if val is None:
                        val = dg_data.get(f"ramp_rate_{self.channel_num}")
                    if val is None:
                        val = dg_data.get("ramp_rate")
                    if val is None:
                        val = data.get(f"ramprate{self.channel_num}")
                    if val is None:
                        val = data.get(f"ramp_rate_{self.channel_num}")
                    if val is None:
                        val = data.get("ramp_rate")
                    if val is not None:
                        if self._ramp_rate != val:
                            self._ramp_rate = val
                            self._update_ramping()
                        return val
        except Exception:
            pass
        return self._ramp_rate

    def _get_ramp_step(self):
        import os
        import json
        try:
            driver_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_dir = os.path.dirname(driver_dir)
            storage_path = os.path.join(workspace_dir, "value_storage")
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    data = json.load(f)
                    dg_data = data.get("dual_gate", {})
                    val = dg_data.get("ramp_step")
                    if val is None:
                        val = data.get("ramp_step")
                    if val is not None:
                        if self._ramp_step != val:
                            self._ramp_step = val
                            self._update_ramping()
                        return val
        except Exception:
            pass
        return self._ramp_step

    def _set_ramp_rate(self, val):
        self._ramp_rate = validate_positive_ramp_value(val, 'Ramp rate')
        self._update_ramping()
        self._save_param_to_storage(f"ramprate{self.channel_num}", val)

    def _set_ramp_step(self, val):
        self._ramp_step = validate_positive_ramp_value(val, 'Ramp step')
        self._update_ramping()
        self._save_param_to_storage("ramp_step", val)

    def _update_ramping(self):
        # One ramp implementation only. QCoDeS stepping used cached voltages and
        # applied another delay/ramp around RampedSet, including after a stop.
        for param in (self.volt, self.curr):
            param.step = None
            param.inter_delay = 0.0

    def _save_param_to_storage(self, key, value):
        import os
        import json
        try:
            driver_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_dir = os.path.dirname(driver_dir)
            storage_path = os.path.join(workspace_dir, "value_storage")
            
            data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        pass
            if "dual_gate" not in data:
                data["dual_gate"] = {}
            data["dual_gate"][key] = value
            with open(storage_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save {key}={value} to value_storage: {e}")


    def _measure_volt(self) -> float:
        import numpy as np
        if not self._output:
            return 0.0
        if self._source_mode == 'VOLT':
            return self._volt + np.random.normal(0, 1e-5)
        else:
            raw_v = self._curr * 100.0
            v = np.clip(raw_v, -self._volt_limit, self._volt_limit)
            return v + np.random.normal(0, 1e-5)

    def _measure_curr(self) -> float:
        import numpy as np
        if not self._output:
            return 0.0
        if self._source_mode == 'CURR':
            return self._curr + np.random.normal(0, 1e-8)
        else:
            raw_i = self._volt / 100.0
            i = np.clip(raw_i, -self._curr_limit, self._curr_limit)
            return i + np.random.normal(0, 1e-8)




class KeysightB2902B(VisaInstrument):
    """
    QCoDeS driver for the Keysight B2902B Dual-Channel Power Source.
    """
    def __init__(self, name, address, configure_defaults=True, **kwargs):
        self._lock = threading.RLock()
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
            # Bypass PyVISA connection initialization
            Instrument.__init__(self, name, **kwargs)
        else:
            super().__init__(name, address, terminator='\n', **kwargs)
            try:
                self.visa_handle.clear()
            except Exception:
                pass
            # Clear status and error queue to prevent visa timeouts
            self.write('*CLS')
            
        # Add channels
        for ch_num in [1, 2]:
            ch_name = f'ch{ch_num}'
            channel = B2902BChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)

        # Protection is mandatory even on a reconnect that preserves other settings.
        self.enforce_current_compliance()
        if not self.simulated and configure_defaults:
            self.ch1.source_mode('VOLT')
            self.ch2.source_mode('VOLT')
            
        if not self.simulated:
            self.connect_message()
        else:
            print(f"Initialized Keysight B2902B '{name}' in Simulation Mode.")

    def enforce_current_compliance(self):
        with self._lock:
            self.ch1.enforce_current_compliance()
            self.ch2.enforce_current_compliance()

    def snapshot_base(self, update = False, params_to_skip_update = None) -> dict:
        if self.simulated:
            return Instrument.snapshot_base(self, update=update, params_to_skip_update=params_to_skip_update)
        return super().snapshot_base(update=update, params_to_skip_update=params_to_skip_update)
            
    def return_to_local(self):
        if not self.simulated:
            try:
                self.visa_handle.clear()
            except Exception:
                pass
            try:
                self.write('*CLS')
            except Exception:
                pass

    def ask(self, cmd: str) -> str:
        if self.simulated:
            return "0.0"
        with self._lock:
            return super().ask(cmd)

    def write(self, cmd: str) -> None:
        if self.simulated:
            return
        with self._lock:
            super().write(cmd)

    def close(self) -> None:
        if getattr(self, 'simulated', False):
            Instrument.close(self)
        else:
            try:
                self.visa_handle.clear()
            except Exception:
                pass
            try:
                super().close()
            except Exception:
                self.remove_instance(self)

    def get_idn(self) -> dict: #eliminates traceback in simulated mode
        if getattr(self, 'simulated', False):
            return {
                'vendor': 'Keysight Technologies',
                'model': 'B2902B',
                'serial': 'simulated',
                'firmware': '5.0.2430'
            }
        return super().get_idn()




