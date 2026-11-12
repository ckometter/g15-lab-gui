"""Dual-gate sweep outputs: a Keysight channel or a uVoltDAQ DAC output per gate.

V1 is driven by Keysight CH1 or one uVoltDAQ DAC output, V2 by Keysight CH2
or another DAC output. A DAC output is wrapped in ``DacGateChannel``, which
offers the Keysight channel calls the DGS uses, so the sweep's ramp,
pause/resume, stop and final ramp to 0 V logic runs unchanged. The uVoltDAQ
only sources voltage: it has no current measurement, output switch or current
compliance. Its current therefore reads NaN, and its voltage readback is the
DAC setpoint (``GET_DAC``), not a measured voltage.
"""
import math

import numpy as np

from drivers.qpg_uvoltdaq import DAC_RANGE_V, validate_dac_voltage
from gate_safety import (DEFAULT_GATE_RAMP_RATE_V_PER_S, validate_gate_voltage,
                         validate_positive_ramp_value)

DEFAULT_GATE_OUTPUTS = {1: "smu:1", 2: "smu:2"}
DAC_CHANNELS = (0, 1, 2, 3)


def gate_output_options(axis):
    """(key, label) choices for gate V1 or V2. Keysight channels keep their gate."""
    return ([(f"smu:{axis}", f"Keysight CH{axis}")]
            + [(f"dac:{number}", f"uVoltDAQ DAC{number}") for number in DAC_CHANNELS])


def gate_output_label(key):
    kind, _, number = str(key).partition(":")
    return f"Keysight CH{number}" if kind == "smu" else f"uVoltDAQ DAC{number}"


def is_dac_output(key):
    return str(key).startswith("dac:")


def dac_number(key):
    return int(str(key).partition(":")[2])


def normalize_gate_outputs(outputs=None):
    """Return {1: key, 2: key}; reject unknown outputs and one DAC driving both gates."""
    outputs = outputs or {}
    result = {}
    for axis in (1, 2):
        key = outputs.get(axis, outputs.get(str(axis), DEFAULT_GATE_OUTPUTS[axis]))
        if key not in dict(gate_output_options(axis)):
            raise ValueError(f"V{axis}: unknown gate output {key!r}.")
        result[axis] = key
    if result[1] == result[2]:
        raise ValueError(f"V1 and V2 cannot both be driven by {gate_output_label(result[1])}.")
    return result


def uses_keysight(outputs):
    return any(not is_dac_output(key) for key in outputs.values())


def validate_gate_output_targets(v1, v2, outputs):
    """Reject any DAC-driven gate target outside the DAC range, before any write."""
    for axis, values in ((1, v1), (2, v2)):
        if not is_dac_output(outputs[axis]):
            continue
        array = np.asarray(values, dtype=float)
        invalid = ~np.isfinite(array) | (np.abs(array) > DAC_RANGE_V)
        if np.any(invalid):
            raise ValueError(
                f"V{axis} is driven by {gate_output_label(outputs[axis])}: a target of "
                f"{float(array[invalid].flat[0]):g} V is outside its ±{DAC_RANGE_V:g} V range. "
                "Sweep blocked before any output changed.")


class _DacRampedSet:
    """``channel.set_r.volt(target)``, as for a Keysight channel."""

    def __init__(self, gate):
        self._gate = gate

    def volt(self, target, callback=None):
        self._gate.ramp_to(target, callback=callback)


class _NoFrontPanel:
    """The DAC has no front-panel remote lock to release."""

    def return_to_local(self):
        pass


class DacGateChannel:
    """A uVoltDAQ DAC output standing in for a Keysight channel in the DGS.

    Ramps use the DAC driver's protected ramp (steps of at most the gate-safety
    step at this gate's ramp rate, final setpoint confirmed). Its stop flag is
    the DAC channel's own, so a sweep's pause/stop cancels the ramp.
    """

    def __init__(self, dac, number, axis, ramp_rate=DEFAULT_GATE_RAMP_RATE_V_PER_S):
        self.dac = dac
        self.number = int(number)
        self.channel_num = int(axis)  # The gate (1 or 2), as a Keysight channel reports it.
        self.ramp_rate_value = validate_positive_ramp_value(ramp_rate, f"V{axis} ramp rate")
        self.label = f"uVoltDAQ DAC{self.number}"
        self.set_r = _DacRampedSet(self)
        self.parent = _NoFrontPanel()

    @property
    def channel(self):
        return self.dac.dac(self.number)

    @property
    def simulated(self):
        return bool(getattr(self.dac, "simulated", False))

    @property
    def ramp_stop_requested(self):
        return bool(self.channel.ramp_stop_requested)

    @ramp_stop_requested.setter
    def ramp_stop_requested(self, value):
        self.channel.ramp_stop_requested = bool(value)

    @property
    def ramping(self):
        return bool(self.channel.ramping)

    def ramp_to(self, target, callback=None):
        channel = self.channel
        channel.ramp_rate(self.ramp_rate_value)
        channel.ramp_to(target, callback=callback)

    def volt(self):
        return float(self.channel.voltage())

    def latest_volt(self):
        value = self.channel.voltage.get_latest()
        return 0.0 if value is None else float(value)

    def measure_volt(self):
        # The DAC outputs have no voltmeter: this is the confirmed setpoint.
        return self.volt()

    def measure_curr(self):
        return math.nan

    def output(self, *args):
        if args:
            raise RuntimeError(f"{self.label} has no output switch; it always drives its setpoint.")
        return True

    def source_mode(self, *args):
        if args and args[0] != "VOLT":
            raise RuntimeError(f"{self.label} can only source voltage.")
        return "VOLT"

    def validate_setpoint(self):
        """The present setpoint must be readable and within the DAC range."""
        return validate_dac_voltage(self.volt(), self.number)


def find_dac(station):
    dac = getattr(station, "dac", None)
    if dac is None:
        raise RuntimeError("The uVoltDAQ is not initialized. Connect it in the Device Manager.")
    return dac


def resolve_gate_channel(station, key, axis, ramp_rate=DEFAULT_GATE_RAMP_RATE_V_PER_S):
    """The channel driving gate V{axis}: ``smu.ch{axis}`` or a DacGateChannel."""
    if is_dac_output(key):
        return DacGateChannel(find_dac(station), dac_number(key), axis, ramp_rate)
    return getattr(station.smu, f"ch{axis}")


def require_gate_outputs_ready(gate, outputs):
    """Check both gates, then confirm Keysight compliance, before any voltage write.

    ``gate(axis)`` returns the channel driving that gate. Keysight channels must
    be in voltage mode at a valid voltage; DAC setpoints must be readable and
    within range. Only Keysight channels that drive a gate get their current
    compliance confirmed: an unused channel is left untouched.
    """
    for axis in (1, 2):
        channel = gate(axis)
        if is_dac_output(outputs[axis]):
            channel.validate_setpoint()
        else:
            if channel.source_mode() != "VOLT":
                raise RuntimeError(f"Channel {axis} is not in voltage-source mode. Gate sweep blocked.")
            validate_gate_voltage(channel.volt(), axis)
    for axis in (1, 2):
        if not is_dac_output(outputs[axis]):
            gate(axis).enforce_current_compliance()
