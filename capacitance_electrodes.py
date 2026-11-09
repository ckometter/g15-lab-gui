"""Electrode outputs for capacitance measurements: assignment, limits and motion.

Each electrode role (top gate, bottom gate, sample) is wired to one DC output:
a Keysight B2902B channel (protected ramp, compliance, leakage readback), a
uVoltDAQ DAC channel (protected ramp), or "not driven" when an external source
holds it at the fixed voltage. Instruments are resolved from the station at
every use, so a device replaced in the Device Manager is picked up directly.

Moves are coordinated: all electrodes advance together along a straight line
in voltage space in increments no larger than the gate-safety step. Unlike
sequential channel ramps, this cannot transiently put a large voltage across
one dielectric (e.g. Vt -> +5 V before Vb follows from -5 V to +5 V).
"""
import math
import threading
import time

from capacitance_bridge import ELECTRODE_LABELS, ELECTRODES, ElectrodeLimits, interpolate_path, swept_electrodes
from drivers.qpg_uvoltdaq import DAC_RANGE_V, validate_dac_voltage
from gate_safety import (GATE_MAX_ABS_VOLTAGE_V, GATE_MAX_RAMP_STEP_V, GateRampCancelled,
                         validate_gate_voltage, validate_positive_ramp_value)

OUTPUT_CHOICES = (
    ("smu:1", "Keysight CH1"),
    ("smu:2", "Keysight CH2"),
    ("dac:0", "uVoltDAQ DAC0"),
    ("dac:1", "uVoltDAQ DAC1"),
    ("dac:2", "uVoltDAQ DAC2"),
    ("dac:3", "uVoltDAQ DAC3"),
    ("none", "Not driven (held externally)"),
)
OUTPUT_LABELS = dict(OUTPUT_CHOICES)
# The qcodes-qpg C_P wiring: DAC0 top gate, DAC1 sample, DAC3 bottom gate.
DEFAULT_WIRING = {"top": "dac:0", "bottom": "dac:3", "sample": "dac:1"}


class MeasurementStopped(RuntimeError):
    """Stop was requested; outputs hold their last confirmed setpoints."""


class RunControl:
    """Thread-safe stop/pause flags shared by a worker, its ramps and the bridge."""

    def __init__(self):
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._stop_callbacks = []

    def on_stop(self, callback):
        self._stop_callbacks.append(callback)

    def stop(self):
        with self.lock:
            self._stop.set()
            self._resume.set()
        for callback in list(self._stop_callbacks):
            try:
                callback()
            except Exception:
                pass

    def pause(self):
        self._resume.clear()

    def resume(self):
        self._resume.set()

    @property
    def stopped(self):
        return self._stop.is_set()

    @property
    def paused(self):
        return not self._resume.is_set()

    def keep_running(self):
        return not self._stop.is_set()

    def check(self):
        if self._stop.is_set():
            raise MeasurementStopped("Measurement stopped; outputs hold their last setpoints.")

    def wait_if_paused(self, on_pause=None):
        """Block while paused. Returns True if a pause actually happened."""
        if self._resume.is_set():
            self.check()
            return False
        if on_pause is not None:
            on_pause()
        while not self._resume.wait(0.05):
            pass
        self.check()
        return True

    def sleep(self, seconds, on_pause=None):
        """Interruptible settle; a pause restarts the full settling time."""
        seconds = max(0.0, float(seconds))
        while True:
            end = time.monotonic() + seconds
            interrupted = False
            while True:
                self.check()
                if self.paused:
                    self.wait_if_paused(on_pause)
                    interrupted = True
                    break
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.02))
            if not interrupted:
                return


class SmuElectrodeOutput:
    kind = "smu"
    max_abs = GATE_MAX_ABS_VOLTAGE_V
    driven = True

    def __init__(self, station, number):
        self.station = station
        self.number = int(number)
        self.label = f"Keysight CH{self.number}"

    @property
    def instrument(self):
        smu = self.station.components.get("smu")
        if smu is None:
            raise RuntimeError("The Keysight SMU is not initialized.")
        return smu

    @property
    def channel(self):
        return getattr(self.instrument, f"ch{self.number}")

    @property
    def simulated(self):
        return bool(getattr(self.instrument, "simulated", False))

    def preflight(self):
        channel = self.channel
        if channel.source_mode() != "VOLT":
            raise RuntimeError(f"{self.label} is not in voltage-source mode. Measurement blocked.")
        validate_gate_voltage(channel.volt(), self.number)
        if not bool(channel.output()):
            raise RuntimeError(f"{self.label} output is OFF. Enable it in the Dual-Gate tab "
                               "(the enable itself ramps safely) before measuring.")
        channel.enforce_current_compliance()

    def setpoint(self):
        return float(self.channel.volt())

    def ramp_to(self, value, control):
        validate_gate_voltage(value, self.number)
        channel = self.channel
        with control.lock:
            control.check()
            channel.ramp_stop_requested = False
        try:
            channel.set_r.volt(value)
        except GateRampCancelled:
            if control.stopped:
                raise MeasurementStopped(f"{self.label} ramp stopped at its last setpoint.")
            raise
        if control.stopped:
            raise MeasurementStopped(f"{self.label} ramp stopped.")

    def request_stop(self):
        try:
            self.channel.ramp_stop_requested = True
        except Exception:
            pass

    def readback(self):
        """(measured voltage, leakage current); NaN when the output is off."""
        channel = self.channel
        if not bool(channel.output()):
            return math.nan, math.nan
        return float(channel.measure_volt()), float(channel.measure_curr())

    def finish(self):
        try:
            self.instrument.return_to_local()
        except Exception:
            pass


class DacElectrodeOutput:
    kind = "dac"
    max_abs = DAC_RANGE_V
    driven = True

    def __init__(self, station, channel, ramp_rate):
        self.station = station
        self.number = int(channel)
        self.ramp_rate = validate_positive_ramp_value(ramp_rate, "DAC ramp rate")
        self.label = f"uVoltDAQ DAC{self.number}"

    @property
    def instrument(self):
        dac = self.station.components.get("dac")
        if dac is None:
            raise RuntimeError("The uVoltDAQ DAC is not initialized.")
        return dac

    @property
    def channel(self):
        return self.instrument.dac(self.number)

    @property
    def simulated(self):
        return bool(getattr(self.instrument, "simulated", False))

    def preflight(self):
        validate_dac_voltage(self.channel.voltage(), self.number)

    def setpoint(self):
        return float(self.channel.voltage())

    def ramp_to(self, value, control):
        channel = self.channel
        with control.lock:
            control.check()
            channel.ramp_stop_requested = False
            channel.ramp_rate(self.ramp_rate)
        try:
            channel.ramp_to(value, keep_running=control.keep_running)
        except GateRampCancelled:
            if control.stopped:
                raise MeasurementStopped(f"{self.label} ramp stopped at its last setpoint.")
            raise
        control.check()

    def request_stop(self):
        try:
            self.channel.ramp_stop_requested = True
        except Exception:
            pass

    def readback(self):
        return self.setpoint(), math.nan

    def finish(self):
        pass


class ExternalElectrode:
    """An electrode held by another source at the fixed voltage; never written."""
    kind = "none"
    max_abs = math.inf
    driven = False
    simulated = False

    def __init__(self, role, assumed_voltage):
        self.role = role
        self.assumed = float(assumed_voltage)
        self.label = OUTPUT_LABELS["none"]

    def preflight(self):
        pass

    def setpoint(self):
        return self.assumed

    def ramp_to(self, value, control):
        if not math.isclose(float(value), self.assumed, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"{ELECTRODE_LABELS[self.role]} is not driven by this GUI and cannot "
                               f"change from {self.assumed:g} V to {float(value):g} V.")

    def request_stop(self):
        pass

    def readback(self):
        return math.nan, math.nan

    def finish(self):
        pass


def make_output(station, key, *, role, fixed_voltage, dac_ramp_rate):
    if key == "none":
        return ExternalElectrode(role, fixed_voltage)
    kind, _, number = str(key).partition(":")
    if kind == "smu" and number in ("1", "2"):
        return SmuElectrodeOutput(station, int(number))
    if kind == "dac" and number in ("0", "1", "2", "3"):
        return DacElectrodeOutput(station, int(number), dac_ramp_rate)
    raise ValueError(f"Unknown electrode output {key!r}.")


def validate_wiring(wiring, fixed):
    """Swept electrodes must be driven; no output may drive two electrodes."""
    for role in ELECTRODES:
        if wiring.get(role) not in OUTPUT_LABELS:
            raise ValueError(f"{ELECTRODE_LABELS[role]}: choose a valid output.")
    for role in swept_electrodes(fixed):
        if wiring[role] == "none":
            raise ValueError(f"{ELECTRODE_LABELS[role]} follows the sweep and must be driven "
                             "by a Keysight or DAC output.")
    used = [wiring[role] for role in ELECTRODES if wiring[role] != "none"]
    duplicates = sorted({key for key in used if used.count(key) > 1})
    if duplicates:
        raise ValueError(f"{OUTPUT_LABELS[duplicates[0]]} is assigned to more than one electrode.")


class ElectrodeSet:
    """The three electrode outputs of one measurement, always moved together."""

    def __init__(self, station, wiring, *, fixed, fixed_voltage, limits, dac_ramp_rate,
                 max_step=GATE_MAX_RAMP_STEP_V):
        validate_wiring(wiring, fixed)
        self.wiring = dict(wiring)
        self.limits = limits if isinstance(limits, ElectrodeLimits) else ElectrodeLimits(**limits)
        self.max_step = min(float(max_step), GATE_MAX_RAMP_STEP_V)
        self.outputs = {role: make_output(station, wiring[role], role=role,
                                          fixed_voltage=fixed_voltage, dac_ramp_rate=dac_ramp_rate)
                        for role in ELECTRODES}

    def output_caps(self):
        return {role: output.max_abs for role, output in self.outputs.items()}

    def check(self, vt, vb, vs, context="Target"):
        self.limits.check(vt, vb, vs, output_caps=self.output_caps(), context=context)

    def preflight(self):
        for output in self.outputs.values():
            output.preflight()

    def setpoints(self):
        return {role: output.setpoint() for role, output in self.outputs.items()}

    def move_to(self, targets, control, progress=None):
        """Straight-line coordinated ramp to ``targets`` (dict role -> V)."""
        targets = {role: float(targets[role]) for role in ELECTRODES}
        self.check(targets["top"], targets["bottom"], targets["sample"])
        start = self.setpoints()
        if all(math.isclose(start[r], targets[r], rel_tol=0, abs_tol=1e-12) for r in ELECTRODES):
            return
        previous = start
        for step in interpolate_path(start, targets, self.max_step):
            control.check()
            for role in ELECTRODES:
                if not math.isclose(step[role], previous[role], rel_tol=0, abs_tol=1e-15):
                    self.outputs[role].ramp_to(step[role], control)
            previous = step
            if progress is not None:
                progress(dict(step))

    def readbacks(self):
        values = {}
        for role, output in self.outputs.items():
            try:
                values[role] = output.readback()
            except Exception:
                values[role] = (math.nan, math.nan)
        return values

    def request_stop(self):
        for output in self.outputs.values():
            output.request_stop()

    def finish(self):
        for output in self.outputs.values():
            output.finish()

    def simulated_modes(self):
        return {role: ("external" if not output.driven else
                       "simulated" if output.simulated else "physical")
                for role, output in self.outputs.items()}
