"""SR830 lock-in wrapper: serialized I/O, coherent X/Y reads, status bits, simulation.

The physical instrument uses the QCoDeS SR830 driver unchanged. The simulated
instrument has the same parameter names and value sets, including the current
inputs (amperes, current sensitivities); its input signal comes from
``signal_source`` (a callable returning the complex input in volts, or amperes
on a current input), which the capacitance and transport tabs bind to their
sample models so measurements can be exercised offline.
"""
import math
import random
import threading

from qcodes.instrument import Instrument
from qcodes.instrument_drivers.stanford_research import SR830
from qcodes.validators import Enum, Numbers

# LIAS? bits (reading the status byte clears latched bits).
STATUS_BITS = {
    0: "input/reserve overload",
    1: "filter overload",
    2: "output overload",
    3: "reference unlock",
}
OVERLOAD_MASK = 0b0111
UNLOCK_MASK = 0b1000

TIME_CONSTANTS = (10e-6, 30e-6, 100e-6, 300e-6, 1e-3, 3e-3, 10e-3, 30e-3, 100e-3, 300e-3,
                  1, 3, 10, 30, 100, 300, 1e3, 3e3, 10e3, 30e3)
FILTER_SLOPES = (6, 12, 18, 24)
VOLTAGE_SENSITIVITIES = tuple(sorted(SR830._VOLT_TO_N))
CURRENT_SENSITIVITIES = tuple(sorted(SR830._CURR_TO_N))
# In these input modes X, Y, R and the sensitivity are currents in amperes.
CURRENT_INPUTS = ("I 1M", "I 100M")


def describe_status(status):
    """Human-readable list of the set LIAS? bits."""
    status = int(status)
    return [text for bit, text in STATUS_BITS.items() if status & (1 << bit)]


def lockin_wait_time(time_constant, filter_slope):
    """Original ``wait_time``: 5/7/9/10 time constants for 6/12/18/24 dB/oct."""
    factor = {6: 5, 12: 7, 18: 9}.get(int(round(float(filter_slope))), 10)
    return factor * float(time_constant)


class StanfordSR830(SR830):
    """SR830 with a simulation mode compatible with the dashboard's device manager."""

    def __init__(self, name: str, address: str, **kwargs):
        self._lock = threading.RLock()
        self.simulated = address == "simulated"
        self.signal_source = None
        if not self.simulated:
            super().__init__(name, address, **kwargs)
            return
        Instrument.__init__(self, name)
        self._sim = {
            "phase": 0.0, "reference_source": "external", "frequency": 7777.7,
            "harmonic": 1, "amplitude": 0.004, "input_config": "a",
            "input_coupling": "AC", "reserve": "normal", "time_constant": 0.3,
            "filter_slope": 12, "sensitivity": 1e-3,
        }
        self._sim_status = 0

        def simulated_parameter(name, label, unit="", vals=None):
            self.add_parameter(name, label=label, unit=unit, vals=vals,
                               get_cmd=lambda key=name: self._sim[key],
                               set_cmd=lambda value, key=name: self._sim.__setitem__(key, value))

        simulated_parameter("phase", "Phase", "deg", Numbers(-360, 729.99))
        simulated_parameter("reference_source", "Reference source", vals=Enum("external", "internal"))
        simulated_parameter("frequency", "Frequency", "Hz", Numbers(1e-3, 102e3))
        simulated_parameter("harmonic", "Harmonic")
        simulated_parameter("amplitude", "Amplitude", "V", Numbers(0.004, 5.0))
        self.add_parameter("input_config", label="Input configuration", vals=Enum("a", "a-b", *CURRENT_INPUTS),
                           get_cmd=lambda: self._sim["input_config"], set_cmd=self._set_simulated_input)
        simulated_parameter("input_coupling", "Input coupling", vals=Enum("AC", "DC"))
        simulated_parameter("reserve", "Reserve", vals=Enum("high", "normal", "low noise"))
        simulated_parameter("time_constant", "Time constant", "s", Enum(*TIME_CONSTANTS))
        simulated_parameter("filter_slope", "Filter slope", "dB/oct", Enum(*FILTER_SLOPES))
        simulated_parameter("sensitivity", "Sensitivity", "V", Enum(*VOLTAGE_SENSITIVITIES))
        self.add_parameter("X", unit="V", get_cmd=lambda: self.snap_xy()[0])
        self.add_parameter("Y", unit="V", get_cmd=lambda: self.snap_xy()[1])
        self.add_parameter("R", unit="V", get_cmd=lambda: math.hypot(*self.snap_xy()))
        self.add_parameter("P", unit="deg", get_cmd=lambda: math.degrees(math.atan2(*reversed(self.snap_xy()))))
        print(f"Initialized Stanford SR830 '{name}' in Simulation Mode.")

    def _set_simulated_input(self, value):
        """As on the instrument: a current input reads amperes, keeping the sensitivity index."""
        was_current = self._sim["input_config"] in CURRENT_INPUTS
        now_current = value in CURRENT_INPUTS
        if was_current != now_current:
            index = (SR830._CURR_TO_N if was_current else SR830._VOLT_TO_N)[self._sim["sensitivity"]]
            self._sim["sensitivity"] = (SR830._N_TO_CURR if now_current else SR830._N_TO_VOLT)[index]
        self._sim["input_config"] = value
        self.sensitivity.vals = Enum(*(CURRENT_SENSITIVITIES if now_current else VOLTAGE_SENSITIVITIES))
        for name in ("X", "Y", "R", "sensitivity"):
            self.parameters[name].unit = "A" if now_current else "V"

    # -- serialized I/O ------------------------------------------------------
    def write_raw(self, cmd: str) -> None:
        if self.simulated:
            return
        with self._lock:
            super().write_raw(cmd)

    def ask_raw(self, cmd: str) -> str:
        if self.simulated:
            raise RuntimeError("Raw SCPI queries are unavailable for the simulated lock-in.")
        with self._lock:
            return super().ask_raw(cmd)

    # -- measurement helpers -------------------------------------------------
    def _simulated_input(self):
        source = self.signal_source
        if source is None:
            return complex(random.gauss(0, 5e-9), random.gauss(0, 5e-9))
        return complex(source(self._sim["time_constant"]))

    def snap_xy(self):
        """Coherent (X, Y) in volts, sampled at one instant."""
        if not self.simulated:
            with self._lock:
                x, y = self.snap("x", "y")
            return float(x), float(y)
        value = self._simulated_input()
        sensitivity = float(self._sim["sensitivity"])
        x, y = value.real, value.imag
        if abs(x) > sensitivity or abs(y) > sensitivity:
            self._sim_status |= 0b0100
            limit = 1.09 * sensitivity
            x, y = max(-limit, min(limit, x)), max(-limit, min(limit, y))
        return x, y

    def status_byte(self):
        """Read (and clear) the LIA status byte; see ``STATUS_BITS``."""
        if self.simulated:
            status, self._sim_status = self._sim_status, 0
            return status
        with self._lock:
            return int(float(self.ask("LIAS?")))

    def clear_status(self):
        self.status_byte()

    def autorange(self, max_changes: int = 1) -> None:
        """QCoDeS autorange; the simulation has no filter, so it does not sleep."""
        if not self.simulated:
            with self._lock:
                return super().autorange(max_changes)
        changes = 0
        while changes <= max_changes:
            r = abs(self._simulated_input())
            sensitivity = float(self._sim["sensitivity"])
            if r > 0.9 * sensitivity:
                changed = self.increment_sensitivity()
            elif r < 0.1 * sensitivity:
                changed = self.decrement_sensitivity()
            else:
                changed = False
            if not changed:
                break
            changes += 1
        self._sim_status = 0

    def get_idn(self) -> dict:
        if self.simulated:
            return {"vendor": "Stanford_Research_Systems", "model": "SR830",
                    "serial": "simulated", "firmware": "sim"}
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
