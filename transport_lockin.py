"""Four-probe transport with two SR830 lock-ins: conversions, checks and a simulated sample.

Ports the qcodes-qpg ``transport_measurement_v2.py`` workflow: lock-in
``lia_I`` reads the sample current, ``lia_V`` the four-probe voltage, and
``R = V_x / I_x`` is mapped over two gate voltages (``transport_measurement.py``
also records the two-terminal resistance). The module has no Qt dependency,
so it is tested offline and can be reused by scripts.

Conventions
-----------
* The current lock-in's sine output excites the sample (through a divider
  and/or a series resistor). Its X and Y are the sample current, either read
  directly on an SR830 current input (I(1M), I(100M): amperes) or as the
  output voltage of a current preamplifier with a sensitivity in A/V.
* The voltage lock-in reads the inner-probe voltage, optionally after a
  voltage preamplifier with gain G; it must be phase-locked to the excitation
  (External reference from the current lock-in's TTL/REF OUT).
* ``R4 = V_x / I_x`` (NaN where |I_x| <= 1e-30 A, as in the script) and
  ``R2 = V_exc / I_x - R_series``, with ``V_exc`` = sine-out amplitude x divider.
"""
import math

import numpy as np

from drivers.stanford_sr830 import CURRENT_INPUTS, CURRENT_SENSITIVITIES, TIME_CONSTANTS, VOLTAGE_SENSITIVITIES
from electrode_sweep import density_field

LOCKINS = ("lockin", "lockin2")
LOCKIN_LABELS = {"lockin": "Lock-in 1", "lockin2": "Lock-in 2"}
CURRENT_MODES = {
    "direct": "SR830 current input (reads A)",
    "preamp": "Current preamp into voltage input",
}
# The script's guard: R = vx / ix if abs(ix) > 1e-30 else nan.
MIN_CURRENT_A = 1e-30
# SR830 sine output (V rms) and reference frequency ranges.
AMPLITUDE_RANGE_V = (0.004, 5.0)
FREQUENCY_RANGE_HZ = (1e-3, 102e3)


def lockin_roles(current_lockin):
    """``{"current": name, "voltage": other}`` for the lock-in chosen to read the current."""
    if current_lockin not in LOCKINS:
        raise ValueError(f"Unknown lock-in {current_lockin!r}; expected one of {LOCKINS}.")
    other = LOCKINS[1] if current_lockin == LOCKINS[0] else LOCKINS[0]
    return {"current": current_lockin, "voltage": other}


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return value


def validate_transport_settings(settings):
    """Hardware-free checks of the lock-in settings; raises ValueError naming the setting."""
    lockin_roles(settings["current_lockin"])
    if settings["current_mode"] not in CURRENT_MODES:
        raise ValueError(f"Unknown current input {settings['current_mode']!r}.")
    if settings["current_mode"] == "preamp":
        _positive(settings["preamp_sensitivity"], "The current preamp sensitivity (A/V)")
    _positive(settings["voltage_gain"], "The voltage preamp gain")
    _positive(settings["excitation_divider"], "The excitation divider")
    series = float(settings["series_resistance"])
    if not math.isfinite(series) or series < 0:
        raise ValueError("The series resistor must be zero or positive.")
    settle = float(settings["settle_tc"])
    if not math.isfinite(settle) or settle < 0:
        raise ValueError("The point settle in time constants must be zero or positive.")
    if settings["time_constant"] != "keep" and not any(
            math.isclose(float(settings["time_constant"]), t, rel_tol=1e-9) for t in TIME_CONSTANTS):
        raise ValueError(f"{settings['time_constant']} s is not an SR830 time constant.")
    if settings["configure_excitation"]:
        low, high = FREQUENCY_RANGE_HZ
        if not low <= float(settings["frequency"]) <= high:
            raise ValueError(f"The excitation frequency must be within {low:g} to {high:g} Hz.")
        low, high = AMPLITUDE_RANGE_V
        if not low <= float(settings["amplitude"]) <= high:
            raise ValueError(f"The SR830 sine amplitude must be within {low:g} to {high:g} V rms.")


def current_from_reading(value, mode, preamp_sensitivity):
    """Sample current (A) from the current lock-in's reading: amperes, or volts after a preamp."""
    value = np.asarray(value, dtype=float)
    return value if mode == "direct" else value * float(preamp_sensitivity)


def voltage_from_reading(value, gain):
    """Four-probe voltage (V) at the sample from the voltage lock-in's reading."""
    return np.asarray(value, dtype=float) / float(gain)


def four_probe_resistance(v_x, i_x):
    """R4 = V_x / I_x; NaN where the current is (numerically) zero."""
    v_x, i_x = np.asarray(v_x, dtype=float), np.asarray(i_x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(i_x) > MIN_CURRENT_A, v_x / i_x, np.nan)


def two_probe_resistance(excitation_v, i_x, series_resistance=0.0):
    """R2 = V_exc / I_x - R_series: two-terminal, including contacts and leads."""
    i_x = np.asarray(i_x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(i_x) > MIN_CURRENT_A,
                        float(excitation_v) / i_x - float(series_resistance), np.nan)


def point_wait(time_constants, settle_tc, extra=0.0):
    """Seconds waited at every point: settle_tc x the longer time constant + extra (script: 3 tau)."""
    return float(settle_tc) * max(float(t) for t in time_constants) + float(extra)


def sensitivity_for(value, current_input):
    """Smallest SR830 sensitivity above ``value`` (the largest if none is)."""
    table = CURRENT_SENSITIVITIES if current_input else VOLTAGE_SENSITIVITIES
    for sensitivity in table:
        if sensitivity >= value:
            return sensitivity
    return table[-1]


def check_lockin_inputs(current, voltage, settings):
    """Compare both lock-ins' readbacks with the settings, before anything moves.

    ``current`` and ``voltage`` hold ``label``, ``input_config``,
    ``reference_source``, ``frequency`` and ``time_constant``. An input mode
    that would make the numbers meaningless raises RuntimeError; reference
    problems are returned as warnings (they spoil the data, not the sample).
    """
    mode = settings["current_mode"]
    if mode == "direct" and current["input_config"] not in CURRENT_INPUTS:
        raise RuntimeError(
            f"{current['label']} reads the current but is set to voltage input "
            f"'{current['input_config']}'. Switch it to current input I(1M) or I(100M), or choose "
            "'Current preamp into voltage input' and enter the preamp sensitivity. Nothing was moved.")
    if mode == "preamp" and current["input_config"] in CURRENT_INPUTS:
        raise RuntimeError(
            f"{current['label']} is set to current input '{current['input_config']}', but a current "
            "preamp is selected, whose output is a voltage. Set the lock-in to input A, or choose "
            "'SR830 current input'. Nothing was moved.")
    if voltage["input_config"] in CURRENT_INPUTS:
        raise RuntimeError(
            f"{voltage['label']} reads the four-probe voltage but is set to current input "
            f"'{voltage['input_config']}'. Set it to input A or A-B. Nothing was moved.")
    warnings = []
    if voltage["reference_source"] != "external":
        if current["reference_source"] != "external":
            warnings.append(
                f"Both lock-ins use their internal oscillators, which are not phase-locked, so the "
                f"voltage phase drifts. Connect {current['label']} TTL/REF OUT to {voltage['label']} "
                "REF IN and select External reference.")
        else:
            warnings.append(f"{voltage['label']} uses its internal reference; lock it to the excitation "
                            "with External reference.")
    f_current, f_voltage = float(current["frequency"]), float(voltage["frequency"])
    if f_current > 0 and abs(f_voltage - f_current) > 0.01 * f_current:
        warnings.append(f"{voltage['label']} reads {f_voltage:.6g} Hz but {current['label']} "
                        f"{f_current:.6g} Hz: the voltage lock-in is not following the excitation.")
    if not math.isclose(float(current["time_constant"]), float(voltage["time_constant"]), rel_tol=1e-6):
        warnings.append(f"The time constants differ ({current['time_constant']:g} s and "
                        f"{voltage['time_constant']:g} s); the settle uses the longer one.")
    return warnings


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulatedTransportSample:
    """Smooth four-probe response of a dual-gated device (bilayer-graphene-like).

    R4 peaks at charge neutrality (n0 = 0), and the peak grows with |p0| as the
    displacement field opens a gap; R2 adds contacts and leads. Values are
    chosen so a 4 mV excitation stays within the SR830's 1 uA current range.
    """

    def __init__(self, r_background=500.0, r_peak=6e3, width=0.4, field_scale=1.5, contacts=4e3, delta=0.0):
        self.r_background = float(r_background)
        self.r_peak = float(r_peak)
        self.width = float(width)
        self.field_scale = float(field_scale)
        self.contacts = float(contacts)
        self.delta = float(delta)

    def r4(self, vt, vb, vs):
        n0, p0 = density_field(vt, vb, vs, delta=self.delta)
        peak = self.r_peak * (1.0 + (p0 / self.field_scale) ** 2)
        return self.r_background + peak / (1.0 + (n0 / self.width) ** 2)

    def r2(self, vt, vb, vs):
        return 2.0 * self.r4(vt, vb, vs) + self.contacts


class SimulatedTransportSignals:
    """Inputs of the two simulated lock-ins measuring one simulated sample.

    The sine output ``excitation()`` (V rms) times the divider drives
    R_series + R2; the voltage probes see I x R4. Both signals share a small
    phase lag, so Y is non-zero while X carries the in-phase resistance.
    ``electrodes()`` returns (Vt, Vb, Vs).
    """

    def __init__(self, sample, excitation, electrodes, settings, *, phase_deg=2.0, noise=0.0, seed=None):
        self.sample = sample
        self._excitation = excitation
        self._electrodes = electrodes
        self.mode = settings["current_mode"]
        self.preamp = float(settings["preamp_sensitivity"])
        self.gain = float(settings["voltage_gain"])
        self.divider = float(settings["excitation_divider"])
        self.series = float(settings["series_resistance"])
        self.rotation = complex(math.cos(math.radians(phase_deg)), -math.sin(math.radians(phase_deg)))
        self.noise = float(noise)  # relative noise at a 0.1 s time constant
        self._rng = np.random.default_rng(seed)

    def currents(self, vt, vb, vs, excitation=None):
        v_exc = (self._excitation() if excitation is None else excitation) * self.divider
        current = v_exc / (self.series + self.sample.r2(vt, vb, vs))
        return current, current * self.sample.r4(vt, vb, vs)

    def _noisy(self, value, time_constant):
        if self.noise > 0:
            scale = self.noise * abs(value) * math.sqrt(0.1 / max(float(time_constant), 1e-5))
            value += complex(*self._rng.normal(0.0, scale, 2))
        return value

    def current_input(self, time_constant=0.1):
        current, _ = self.currents(*self._electrodes())
        reading = float(current) if self.mode == "direct" else float(current) / self.preamp
        return self._noisy(reading * self.rotation, time_constant)

    def voltage_input(self, time_constant=0.1):
        _, voltage = self.currents(*self._electrodes())
        return self._noisy(float(voltage) * self.gain * self.rotation, time_constant)

    def peak_readings(self, vt, vb, vs, excitation=None):
        """Largest |reading| of each lock-in over the given targets (for the simulated ranges)."""
        current, voltage = self.currents(np.asarray(vt), np.asarray(vb), np.asarray(vs), excitation)
        current_reading = current if self.mode == "direct" else current / self.preamp
        return float(np.nanmax(np.abs(current_reading))), float(np.nanmax(np.abs(voltage * self.gain)))
