"""QPG uVoltDAQ 4O4I (four DAC outputs, four ADC inputs) with protected ramps.

Command set as in qcodes-qpg ``QPG_uVoltDAQ_4O4I.py``: ``GET_DAC,ch``,
``SET,ch,value`` and ``GET_ADC,ch``. Unlike that driver, a voltage write never
jumps: every write (``dac0.voltage(v)`` or ``ramp_to``) re-reads the actual
setpoint, moves in increments no larger than the gate-safety step at the
channel's ramp rate, and confirms the final setpoint. ``INITIALIZE`` is never
sent automatically, so existing output voltages are retained on connection.
"""
import math
import random
import threading
import time

from qcodes.instrument import Instrument, InstrumentChannel, VisaInstrument

from gate_safety import (DEFAULT_GATE_RAMP_RATE_V_PER_S, GATE_MAX_RAMP_STEP_V,
                         GateRampCancelled, validate_positive_ramp_value)

DAC_RANGE_V = 10.0
READBACK_TOLERANCE_V = 1e-3


class UVoltDacChannel(InstrumentChannel):
    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        self.channel = channel
        self.ramping = False
        self.ramp_stop_requested = False
        self._ramp_lock = threading.RLock()
        self._ramp_rate = DEFAULT_GATE_RAMP_RATE_V_PER_S
        self._ramp_step = GATE_MAX_RAMP_STEP_V
        self._sim_voltage = 0.0
        self.add_parameter("voltage", label=f"DAC {channel} voltage", unit="V",
                           get_cmd=self._get_voltage, set_cmd=lambda value: self.ramp_to(value))
        self.add_parameter("ramp_rate", label=f"DAC {channel} ramp rate", unit="V/s",
                           get_cmd=lambda: self._ramp_rate, set_cmd=self._set_ramp_rate)
        self.add_parameter("ramp_step", label=f"DAC {channel} ramp step", unit="V",
                           get_cmd=lambda: self._ramp_step, set_cmd=self._set_ramp_step)

    def _set_ramp_rate(self, value):
        self._ramp_rate = validate_positive_ramp_value(value, "DAC ramp rate")

    def _set_ramp_step(self, value):
        value = validate_positive_ramp_value(value, "DAC ramp step")
        self._ramp_step = min(value, GATE_MAX_RAMP_STEP_V)

    def _get_voltage(self):
        if self.root_instrument.simulated:
            return self._sim_voltage
        return float(self.root_instrument.ask(f"GET_DAC,{self.channel}"))

    def _write_step(self, value):
        value = validate_dac_voltage(value, self.channel)
        if self.root_instrument.simulated:
            self._sim_voltage = value
            return
        response = self.root_instrument.ask(f"SET,{self.channel},{value:.9e}")
        if "ERR" in str(response).upper():
            raise RuntimeError(f"DAC {self.channel} rejected SET {value:g} V: {response!r}")

    def _wait(self, seconds, keep_running):
        end = time.monotonic() + seconds
        while True:
            if self.ramp_stop_requested or not keep_running():
                raise GateRampCancelled(f"DAC {self.channel}: ramp stopped at its last setpoint.")
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.02))

    def ramp_to(self, target, *, keep_running=None, callback=None):
        """Ramp from the actual setpoint; cancellation keeps the last written step."""
        keep_running = keep_running or (lambda: True)
        target = validate_dac_voltage(target, self.channel)
        rate = validate_positive_ramp_value(self._ramp_rate, "DAC ramp rate")
        step = min(validate_positive_ramp_value(self._ramp_step, "DAC ramp step"), GATE_MAX_RAMP_STEP_V)
        with self._ramp_lock:
            self.ramping = True
            try:
                start = float(self._get_voltage())
                if not math.isfinite(start):
                    raise RuntimeError(f"DAC {self.channel}: non-finite setpoint readback; ramp blocked.")
                count = math.ceil(abs(target - start) / step - 1e-12)
                previous = start
                for index in range(1, count + 1):
                    value = target if index == count else start + (target - start) * index / count
                    if not self.root_instrument.simulated:
                        self._wait(abs(value - previous) / rate, keep_running)
                    elif self.ramp_stop_requested or not keep_running():
                        raise GateRampCancelled(f"DAC {self.channel}: ramp stopped at its last setpoint.")
                    self._write_step(value)
                    self.voltage.cache.set(value)
                    previous = value
                    if callback is not None:
                        callback(value)
                if count:
                    actual = float(self._get_voltage())
                    if not math.isfinite(actual) or abs(actual - target) > READBACK_TOLERANCE_V:
                        raise RuntimeError(f"DAC {self.channel}: final setpoint not confirmed "
                                           f"(requested {target:g} V, read {actual:g} V). Operation blocked.")
            finally:
                self.ramping = False


class UVoltAdcChannel(InstrumentChannel):
    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        self.channel = channel
        self.add_parameter("voltage", label=f"ADC {channel} voltage", unit="V",
                           get_cmd=self._get_voltage)

    def _get_voltage(self):
        if self.root_instrument.simulated:
            return random.gauss(0.0, 2e-6)
        return float(self.root_instrument.ask(f"GET_ADC,{self.channel}"))


def validate_dac_voltage(value, channel=None):
    value = float(value)
    if not math.isfinite(value) or abs(value) > DAC_RANGE_V:
        label = f"DAC {channel}" if channel is not None else "DAC"
        raise ValueError(f"{label}: voltage must be finite and within ±{DAC_RANGE_V:g} V.")
    return value


class QPGuVoltDAQ(VisaInstrument):
    """Four protected DAC outputs (``dac0``..``dac3``) and four ADC inputs (``adc0``..``adc3``)."""

    def __init__(self, name, address, timeout=5.0, **kwargs):
        self._lock = threading.RLock()
        self.simulated = address == "simulated"
        if self.simulated:
            Instrument.__init__(self, name)
        else:
            kwargs.setdefault("terminator", "\r\n")
            super().__init__(name, address, timeout=timeout, **kwargs)
        for channel in range(4):
            self.add_submodule(f"dac{channel}", UVoltDacChannel(self, f"dac{channel}", channel))
        for channel in range(4):
            self.add_submodule(f"adc{channel}", UVoltAdcChannel(self, f"adc{channel}", channel))
        if self.simulated:
            print(f"Initialized QPG uVoltDAQ '{name}' in Simulation Mode.")
        else:
            self.connect_message()

    def dac(self, channel):
        return getattr(self, f"dac{int(channel)}")

    def adc(self, channel):
        return getattr(self, f"adc{int(channel)}")

    def ask(self, cmd: str) -> str:
        if self.simulated:
            raise RuntimeError("Raw queries are unavailable for the simulated DAC.")
        with self._lock:
            return super().ask(cmd)

    def write(self, cmd: str) -> None:
        if self.simulated:
            return
        with self._lock:
            super().write(cmd)

    def get_idn(self) -> dict:
        if self.simulated:
            return {"vendor": "QPG", "model": "uVoltDAQ 4O4I", "serial": "simulated", "firmware": "sim"}
        return super().get_idn()

    def snapshot_base(self, update=False, params_to_skip_update=None):
        if self.simulated:
            return Instrument.snapshot_base(self, update=update, params_to_skip_update=params_to_skip_update)
        return super().snapshot_base(update=update, params_to_skip_update=params_to_skip_update)

    def close(self) -> None:
        if getattr(self, "simulated", False):
            Instrument.close(self)
        else:
            super().close()
