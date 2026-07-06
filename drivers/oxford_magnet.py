# -*- coding: utf-8 -*-
"""
QCoDeS driver for the Spectromag Oxford Magnet.
Provides compatibility with legacy Daemondaq drivers and native QCoDeS integration.
"""
import time
import logging
import math
import numpy as np
import threading

from qcodes.instrument import VisaInstrument, Instrument
from qcodes import validators as vals

log = logging.getLogger(__name__)

class OxfordMagnet(VisaInstrument):
    """
    QCoDeS driver wrapper for the Spectromag Oxford Magnet.
    Supports physical GPIB VISA connections and simulated ramping.
    """
    MIN_SWITCH_HEATER_WAIT_TIME = 1.0
    MAX_SWEEP_RATE_HIGH_B = 1.05
    MAX_SWEEP_RATE_LOW_B = 1.05
    MIDPOINT_B = 8  # Tesla
    SWITCH_HEATER_FITTED = False
    SLEEPTIME = 0.1
    TIMEOUT = 10000

    def __init__(self, name, address, visalib='default', output_handle=None, **kwargs):
        self._lock = threading.RLock()
        # Clean up any existing instances
        try:
            instr = Instrument.find_instrument(name)
            try: Instrument.remove_instance(instr)
            except Exception: pass
            try: instr.close()
            except Exception: pass
        except KeyError:
            pass

        self.simulated = (address == 'simulated' or visalib == 'simulated')
        self.visa_address = address
        self.output_handle = output_handle if output_handle is not None else import_sys_stdout()

        # Ramping simulation states
        self._sim_current_field = 0.0
        self._sim_target_field = 0.0
        self._sim_sweep_rate = 1.0  # T/min
        self._sim_switch_heater = 'off'
        self._sim_persistent_field = 0.0
        self._sim_activity = 'hold'
        self._sim_start_time = time.time()
        self._sim_start_field = 0.0
        self._sim_sweep_duration = 0.0

        if self.simulated:
            Instrument.__init__(self, name, **kwargs)
        else:
            # Connect physically using VisaInstrument
            super().__init__(name, address, terminator='\r', **kwargs)
            self.visa_handle.timeout = self.TIMEOUT
            self.write('Q0')  # set normal resolution and \r termination

        # Add parameters
        self.add_parameter('field',
                           label='Magnetic Field',
                           unit='T',
                           get_cmd=self.get_field,
                           set_cmd=self.set_field,
                           vals=vals.Numbers(-15.0, 15.0))

        self.add_parameter('output_field',
                           label='Output Field',
                           unit='T',
                           get_cmd=self._get_output_field)

        self.add_parameter('persistent_magnet_field',
                           label='Persistent Magnet Field',
                           unit='T',
                           get_cmd=self._get_persistent_magnet_field)

        self.add_parameter('target_field_param',
                           label='Target Field Setpoint',
                           unit='T',
                           get_cmd=self._get_target_field,
                           set_cmd=self._set_target_field)

        self.add_parameter('field_sweep_rate',
                           label='Field Sweep Rate',
                           unit='T/min',
                           get_cmd=self._get_field_sweep_rate,
                           set_cmd=self._set_field_sweep_rate)

        self.add_parameter('switch_heater',
                           label='Switch Heater',
                           get_cmd=self._get_switch_heater,
                           set_cmd=self._set_switch_heater)

        self.add_parameter('activity',
                           label='Activity',
                           get_cmd=self._get_activity)

        self.add_parameter('is_sweeping',
                           label='Is Sweeping',
                           get_cmd=self._get_is_sweeping)

    def get_idn(self) -> dict:
        if getattr(self, 'simulated', False):
            return {
                'vendor': 'Oxford Instruments',
                'model': 'Spectromag',
                'serial': 'simulated',
                'firmware': '1.0'
            }
        return super().get_idn()

    def snapshot_base(self, update=False, params_to_skip_update=None) -> dict:
        if getattr(self, 'simulated', False):
            return Instrument.snapshot_base(self, update=update, params_to_skip_update=params_to_skip_update)
        return super().snapshot_base(update=update, params_to_skip_update=params_to_skip_update)

    def ask(self, cmd: str) -> str:
        if getattr(self, 'simulated', False):
            return "0.0"
        with self._lock:
            return super().ask(cmd)

    def write(self, cmd: str) -> None:
        if getattr(self, 'simulated', False):
            return
        with self._lock:
            super().write(cmd)

    def close(self) -> None:
        if getattr(self, 'simulated', False):
            Instrument.close(self)
        else:
            try:
                super().close()
            except Exception:
                self.remove_instance(self)

    # Simulated vs Visa communication helpers
    def _query(self, cmd):
        if self.simulated:
            return "0.0"
        with self._lock:
            time.sleep(self.SLEEPTIME)
            try:
                res = self.visa_handle.query(cmd)
            except Exception as e:
                try:
                    self.visa_handle.clear()
                except Exception:
                    pass
                raise e
            if res.startswith('?'):
                raise RuntimeError('Error raised by magnet: ' + res)
            return res

    def _write(self, cmd):
        if self.simulated:
            return
        with self._lock:
            time.sleep(self.SLEEPTIME)
            try:
                self.visa_handle.write(cmd)
                res = self.visa_handle.read()
            except Exception as e:
                try:
                    self.visa_handle.clear()
                except Exception:
                    pass
                raise e
            if res.startswith('?'):
                raise RuntimeError('Error raised by magnet: ' + res)

    # Parameter getter/setter methods
    def _get_output_field(self) -> float:
        if self.simulated:
            self._update_simulated_sweep()
            return self._sim_current_field
        res = self._query('R7')
        return float(res[1:])

    def _get_persistent_magnet_field(self) -> float:
        if self.simulated:
            return self._sim_persistent_field
        res = self._query('R18')
        return float(res[1:])

    def _get_target_field(self) -> float:
        if self.simulated:
            return self._sim_target_field
        res = self._query('R8')
        return float(res[1:])

    def _set_target_field(self, val: float) -> None:
        if self.simulated:
            self._sim_target_field = val
            return
        self._write(f'J{val}')

    def _get_field_sweep_rate(self) -> float:
        if self.simulated:
            return self._sim_sweep_rate
        res = self._query('R9')
        return float(res[1:])

    def _set_field_sweep_rate(self, val: float) -> None:
        if self.simulated:
            self._sim_sweep_rate = val
            return
        self._write(f'T{val}')

    def _get_switch_heater(self) -> str:
        if self.simulated:
            return self._sim_switch_heater
        if not self.SWITCH_HEATER_FITTED:
            return 'no switch heater'
        res = self._query('X')
        state = int(res[8])
        return {0: 'off (magnet at zero)',
                1: 'on',
                2: 'off (magnet at field)',
                4: 'fault (heater off, switch open)',
                5: 'fault (heater on, switch closed)'}.get(state, 'unknown')

    def _set_switch_heater(self, val: str) -> None:
        if self.simulated:
            self._sim_switch_heater = val
            return
        if not self.SWITCH_HEATER_FITTED:
            return
        map_ = {'off': 0, 'on': 1}
        if val in ['on', 'off']:
            self._write(f'H{map_[val]}')
            self.output_handle.write('Changing switch heater... ')
            time.sleep(self.MIN_SWITCH_HEATER_WAIT_TIME)
            t0 = time.time()
            while (not self._get_switch_heater().startswith(val) and
                   time.time() <= t0 + self.TIMEOUT):
                time.sleep(.1)
            self.output_handle.write('Done.\n')
            if time.time() > t0 + self.TIMEOUT:
                log.warning('Wait while changing switch heater timed out.')
        else:
            raise ValueError("Value must be 'on' or 'off'.")

    def _get_activity(self) -> str:
        if self.simulated:
            self._update_simulated_sweep()
            return self._sim_activity
        res = self._query('X')
        pos4 = int(res[4])
        return {0: 'hold', 1: 'to set point', 2: 'to zero', 4: 'clamped'}.get(pos4, 'unknown')

    def _get_is_sweeping(self) -> bool:
        if self.simulated:
            self._update_simulated_sweep()
            return self._sim_activity in ['to set point', 'to zero']
        res = self._query('X')
        val = int(res[11])
        if val == 0:
            return False
        elif val in [1, 2, 3]:
            return True
        else:
            raise IOError(f'Unexpected output from device: {res}')

    def hold(self):
        if self.simulated:
            self._update_simulated_sweep()
            self._sim_activity = 'hold'
            return
        self._write('A0')

    def goto_zero(self):
        if self.simulated:
            self._update_simulated_sweep()
            self._sim_start_field = self._sim_current_field
            self._sim_target_field = 0.0
            self._sim_activity = 'to zero'
            self._sim_start_time = time.time()
            rate = self._sim_sweep_rate / 60.0  # T/s
            self._sim_sweep_duration = abs(self._sim_start_field) / rate if rate > 0 else 0.0
            return
        self._write('A2')

    def goto_set(self):
        if self.simulated:
            self._update_simulated_sweep()
            self._sim_start_field = self._sim_current_field
            self._sim_activity = 'to set point'
            self._sim_start_time = time.time()
            rate = self._sim_sweep_rate / 60.0  # T/s
            dist = abs(self._sim_target_field - self._sim_start_field)
            self._sim_sweep_duration = dist / rate if rate > 0 else 0.0
            return
        self._write('A1')

    def clamp(self):
        if self.simulated:
            self._sim_activity = 'clamped'
            return
        self._write('A4')

    # Simulation helper to update B-field over time
    def _update_simulated_sweep(self):
        if self._sim_activity == 'to set point':
            dt = (time.time() - self._sim_start_time) * 30.0
            if self._sim_sweep_duration <= 0 or dt >= self._sim_sweep_duration:
                self._sim_current_field = self._sim_target_field
                self._sim_activity = 'hold'
            else:
                fraction = dt / self._sim_sweep_duration
                self._sim_current_field = self._sim_start_field + fraction * (self._sim_target_field - self._sim_start_field)
            if self._sim_switch_heater == 'on':
                self._sim_persistent_field = self._sim_current_field
        elif self._sim_activity == 'to zero':
            dt = (time.time() - self._sim_start_time) * 30.0
            if self._sim_sweep_duration <= 0 or dt >= self._sim_sweep_duration:
                self._sim_current_field = 0.0
                self._sim_activity = 'hold'
            else:
                fraction = dt / self._sim_sweep_duration
                self._sim_current_field = self._sim_start_field + fraction * (0.0 - self._sim_start_field)
            if self._sim_switch_heater == 'on':
                self._sim_persistent_field = self._sim_current_field

    # Legacy Daemondaq compatibility functions
    def get_field(self) -> float:
        return self._get_output_field()

    def set_field(self, field: float) -> None:
        self.goto_field(field)

    def take_leads_to_field(self, field: float) -> None:
        """Ramp the current leads to the specified field value with the switch heater off."""
        if isinstance(field, int):
            field = float(field)
        field = np.round(field, 4)
        self._set_target_field(field)
        self.goto_set()
        time.sleep(0.5)
        t0 = time.time()
        while self._get_is_sweeping() and time.time() <= t0 + self.TIMEOUT:
            time.sleep(0.1)
        self.hold()

    def goto_field(self, field: float, sweep_rate='default',
                   take_leads_to_zero: bool=False,
                   disable_switch_heater: bool=True):
        if isinstance(field, int):
            field = float(field)
        field = np.round(field, 4)

        if sweep_rate == 'default':
            sweep_rate = self.MAX_SWEEP_RATE_LOW_B

        self.field_sweep_rate(sweep_rate)

        # Check switch heater state and match leads if needed
        if self.SWITCH_HEATER_FITTED:
            heater = self._get_switch_heater()
            if heater.startswith('off'):
                output_f = self._get_output_field()
                persistent_f = self._get_persistent_magnet_field()
                # If leads do not match coil field, match them first
                if not np.isclose(output_f, persistent_f, atol=0.01):
                    self.take_leads_to_field(persistent_f)
                self._set_switch_heater('on')

        # Ramp to target
        self._set_target_field(field)
        self.goto_set()

        # Block until sweep is complete
        time.sleep(0.1)
        t0 = time.time()
        while self._get_is_sweeping() and time.time() <= t0 + self.TIMEOUT:
            time.sleep(0.1)

        self.hold()

        # Disable heater and take leads to zero if configured
        if self.SWITCH_HEATER_FITTED:
            if disable_switch_heater:
                self._set_switch_heater('off')
                if take_leads_to_zero:
                    self.take_leads_to_field(0.0)

    def print_system_overview(self):
        self.output_handle.write(
            f"output field: {self._get_output_field():.4f} T\n"
            f"target field: {self._get_target_field():.4f} T\n"
            f"sweep rate:   {self._get_field_sweep_rate():.4f} T/min\n"
            f"activity:     {self._get_activity()}\n"
            f"heater:       {self._get_switch_heater()}\n"
        )


def import_sys_stdout():
    import sys
    return sys.stdout
