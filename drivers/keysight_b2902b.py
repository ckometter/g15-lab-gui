import threading
from qcodes.instrument import VisaInstrument, Instrument
from qcodes.instrument import InstrumentChannel

class RampedSet:
    """
    Helper class to perform software parameter ramping on a channel.
    """
    def __init__(self, channel):
        self.channel = channel


    def _ramp(self, parameter_name: str, target_val: float, callback=None):
        self.channel.ramping = True
        try:
            self._do_ramp(parameter_name, target_val, callback=callback)
        finally:
            self.channel.ramping = False

    def _do_ramp(self, parameter_name: str, target_val: float, callback=None):
        import os
        import json
        import time
        
        field_rate = f"ramprate{self.channel.channel_num}"
        field_step = "ramp_step"
        
        driver_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(driver_dir)
        storage_path = os.path.join(workspace_dir, "value_storage")
        with open(storage_path, "r") as f:
            data = json.load(f)
        
        dg_data = data.get("dual_gate", {})
        ramp_rate = dg_data.get(field_rate)
        if ramp_rate is None:
            ramp_rate = dg_data.get(f"ramp_rate_{self.channel.channel_num}")
        if ramp_rate is None:
            ramp_rate = dg_data.get("ramp_rate")
        if ramp_rate is None:
            ramp_rate = data.get(field_rate)
        if ramp_rate is None:
            ramp_rate = data.get(f"ramp_rate_{self.channel.channel_num}")
        if ramp_rate is None:
            ramp_rate = data.get("ramp_rate", 0.1)
            
        ramp_step = dg_data.get(field_step)
        if ramp_step is None:
            ramp_step = data.get(field_step, 0.0005)
        
        param = getattr(self.channel, parameter_name)
        
        if ramp_rate <= 0 or ramp_step <= 0:
            param(target_val)
            if callback is not None:
                try: callback(target_val)
                except Exception: pass
            return
            
        try:
            current_val = param.get_latest()
            if current_val is None:
                current_val = param()
        except Exception:
            current_val = param()
        dval = target_val - current_val
        
        num_steps = int(abs(dval) / ramp_step)
        if num_steps <= 1:
            param(target_val)
            if callback is not None:
                try: callback(target_val)
                except Exception: pass
            return
            
        dt = ramp_step / ramp_rate
        
        # Calculate expected ramping duration
        ramp_time = abs(dval) / ramp_rate
        new_timeout = max(5.0, ramp_time + 10.0)
        
        # Temporarily increase VISA timeout on the parent SMU instrument to avoid warnings
        smu = self.channel.parent
        old_timeout = None
        if hasattr(smu, "timeout") and not getattr(smu, "simulated", False):
            try:
                if hasattr(smu, "_lock"):
                    with smu._lock:
                        old_timeout = smu.timeout()
                        smu.timeout(new_timeout)
                else:
                    old_timeout = smu.timeout()
                    smu.timeout(new_timeout)
            except Exception:
                old_timeout = None

        self.channel.ramp_stop_requested = False
        try:
            for i in range(1, num_steps + 1):
                if getattr(self.channel, "ramp_stop_requested", False):
                    break
                fraction = i / num_steps
                next_val = current_val + dval * fraction
                param(next_val)
                if callback is not None:
                    try: callback(next_val)
                    except Exception: pass
                time.sleep(dt)
                
            if not getattr(self.channel, "ramp_stop_requested", False):
                param(target_val)
                if callback is not None:
                    try: callback(target_val)
                    except Exception: pass
        finally:
            # Restore the original visa timeout
            if old_timeout is not None and hasattr(smu, "timeout") and not getattr(smu, "simulated", False):
                try:
                    if hasattr(smu, "_lock"):
                        with smu._lock:
                            smu.timeout(old_timeout)
                    else:
                        smu.timeout(old_timeout)
                except Exception:
                    pass

    def volt(self, target_voltage: float, callback=None):
        self._ramp("volt", target_voltage, callback=callback)

    def curr(self, target_current: float, callback=None):
        self._ramp("curr", target_current, callback=callback)

class B2902BChannel(InstrumentChannel):
    """
    Represent a single channel of the Keysight B2902B Low Noise Power Source.
    """
    def __init__(self, parent, name, channel_num):
        super().__init__(parent, name)
        self.channel_num = channel_num
        self.simulated = parent.simulated
        self.ramping = False
        self.set_r = RampedSet(self)
        
        if self.simulated:
            self._volt = 0.0
            self._curr = 0.0
            self._output = 0
            self._curr_limit = 0.1
            self._volt_limit = 10.0
            self._source_mode = 'VOLT'
            
            self.add_parameter('volt',
                               label=f'Channel {channel_num} Voltage',
                               unit='V',
                               get_cmd=lambda: self._volt,
                               set_cmd=self._set_volt)
            
            self.add_parameter('curr',
                               label=f'Channel {channel_num} Current',
                               unit='A',
                               get_cmd=lambda: self._curr,
                               set_cmd=self._set_curr)
            
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
                               set_cmd=self._set_curr_limit)
            
            self.add_parameter('volt_limit',
                               label=f'Channel {channel_num} Voltage Limit',
                               unit='V',
                               get_cmd=lambda: self._volt_limit,
                               set_cmd=self._set_volt_limit)
            
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
                               set_cmd=f':SOUR{channel_num}:FUNC:MODE {{}}',
                               val_mapping={'VOLT': 'VOLT', 'CURR': 'CURR'})

            # Active voltage sourcing commands
            self.add_parameter('volt',
                               label=f'Channel {channel_num} Voltage',
                               unit='V',
                               get_cmd=f':SOUR{channel_num}:VOLT?',
                               set_cmd=f':SOUR{channel_num}:VOLT {{:.6e}}',
                               get_parser=float)
            
            # Active current sourcing commands
            self.add_parameter('curr',
                               label=f'Channel {channel_num} Current',
                               unit='A',
                               get_cmd=f':SOUR{channel_num}:CURR?',
                               set_cmd=f':SOUR{channel_num}:CURR {{:.6e}}',
                               get_parser=float)
            
            self.add_parameter('output',
                               label=f'Channel {channel_num} Output State',
                               get_cmd=f':OUTP{channel_num}?',
                               set_cmd=f':OUTP{channel_num} {{:d}}',
                               val_mapping={True: 1, False: 0})
            
            # Current compliance limit (used in voltage sourcing mode)
            self.add_parameter('curr_limit',
                               label=f'Channel {channel_num} Current Limit',
                               unit='A',
                               get_cmd=f':SENS{channel_num}:CURR:PROT?',
                               set_cmd=f':SENS{channel_num}:CURR:PROT {{:.6e}}',
                               get_parser=float)

            # Voltage compliance limit (used in current sourcing mode)
            self.add_parameter('volt_limit',
                               label=f'Channel {channel_num} Voltage Limit',
                               unit='V',
                               get_cmd=f':SENS{channel_num}:VOLT:PROT?',
                               set_cmd=f':SENS{channel_num}:VOLT:PROT {{:.6e}}',
                               get_parser=float)
            
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
        self._ramp_rate = None
        self._ramp_step = None
        
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
        self._volt = val

    def _set_curr(self, val):
        self._curr = val

    def _set_source_mode(self, val):
        self._source_mode = val

    def _set_output(self, val):
        self._output = int(val)

    def _set_curr_limit(self, val):
        self._curr_limit = val

    def _set_volt_limit(self, val):
        self._volt_limit = val

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
        self._ramp_rate = val
        self._update_ramping()
        self._save_param_to_storage(f"ramprate{self.channel_num}", val)

    def _set_ramp_step(self, val):
        self._ramp_step = val
        self._update_ramping()
        self._save_param_to_storage("ramp_step", val)

    def _update_ramping(self):
        if self._ramp_step is not None and self._ramp_rate is not None and self._ramp_rate > 0:
            self.volt.step = self._ramp_step
            self.volt.inter_delay = self._ramp_step / self._ramp_rate
            self.curr.step = self._ramp_step
            self.curr.inter_delay = self._ramp_step / self._ramp_rate
        else:
            self.volt.step = None
            self.volt.inter_delay = 0.0
            self.curr.step = None
            self.curr.inter_delay = 0.0

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
    def __init__(self, name, address, **kwargs):
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
            # Initialize source mode to VOLT for both channels
            self.write(':SOUR1:FUNC:MODE VOLT')
            self.write(':SOUR2:FUNC:MODE VOLT')
            
        # Add channels
        for ch_num in [1, 2]:
            ch_name = f'ch{ch_num}'
            channel = B2902BChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            
        if not self.simulated:
            self.connect_message()
        else:
            print(f"Initialized Keysight B2902B '{name}' in Simulation Mode.")

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




