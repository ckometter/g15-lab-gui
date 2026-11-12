"""Capacitance-bridge balancing for the capacitance tab.

Ported from the qcodes-qpg ``capacitance_measurement.py`` script and its
``include/capacitance_bridge.py`` (``CapacitanceBridgeSR830Lockin``). The module
has no Qt dependency so the algorithm can be tested offline and reused by
scripts. The bridge reports the sample capacitance and dissipation as ratios to
the reference capacitor, ``C/C_ref`` and ``D/C_ref``; multiply by ``C_ref`` for
farads.

The electrode coordinates (n0, p0), sweep plans and voltage limits are shared
with the transport tab and live in ``electrode_sweep``; they are re-exported
here so existing scripts keep working.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from drivers.stanford_sr830 import lockin_wait_time  # noqa: F401  (re-exported)
from electrode_sweep import (  # noqa: F401  (re-exported for scripts)
    COORDINATE_MODES, ELECTRODE_LABELS, ELECTRODE_SYMBOLS, ELECTRODES, ElectrodeLimits, SweepPlan,
    axis_values, build_sweep_plan, coordinate_names, density_field, direct_voltages, electrode_voltages,
    estimate_duration, interpolate_path, plan_summary, swept_electrodes, transform_description,
    validate_delta, validate_electrode)

# Measurement presets: AC excitation and measurement node are wiring facts that
# are recorded with the data; the preset only chooses the default DC-fixed
# electrode (the measurement node, whose DC level the bridge amplifier sets).
MEASUREMENT_TYPES = {
    "penetration": {
        "label": "Penetration field capacitance C_P",
        "symbol": "C_P",
        "fixed": "bottom",
        "excited": "Top gate",
        "measured": "Bottom gate",
    },
    "top_gate": {
        "label": "Top-gate capacitance C_T",
        "symbol": "C_T",
        "fixed": "sample",
        "excited": "Top gate",
        "measured": "Sample",
    },
    "bottom_gate": {
        "label": "Bottom-gate capacitance C_B",
        "symbol": "C_B",
        "fixed": "sample",
        "excited": "Bottom gate",
        "measured": "Sample",
    },
    "symmetric": {
        "label": "Symmetric capacitance C_S",
        "symbol": "C_S",
        "fixed": "sample",
        "excited": "Both gates",
        "measured": "Sample",
    },
    "custom": {
        "label": "Custom",
        "symbol": "C",
        "fixed": "sample",
        "excited": "(record in comment)",
        "measured": "(record in comment)",
    },
}

# Siglent SDG2042X amplitude range (Vpp, high-Z) used by the balance check.
AWG_MIN_AMPLITUDE_VPP = 2e-3


class BalanceError(RuntimeError):
    """The bridge could not be balanced; no capacitance calibration exists."""


class BalanceCancelled(RuntimeError):
    """Stop was requested while the bridge was being balanced."""


# ---------------------------------------------------------------------------
# Bridge balancing (port of LinearBalancingBridge / CapacitanceBridgeSR830Lockin)
# ---------------------------------------------------------------------------

def ac_scale_factor(ref_atten_db, sample_atten_db, sample_amplitude):
    """Original ``balance()`` scale: 10**(-(ref_atten - sample_atten)/20) / ch1_v."""
    amplitude = float(sample_amplitude)
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("The sample excitation amplitude must be positive.")
    return 10 ** (-(float(ref_atten_db) - float(sample_atten_db)) / 20.0) / amplitude


def vector_phase_deg(vector):
    """Original ``vec_phase``: AWG phase (0-360 deg) for a balance vector."""
    phase = -math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    if phase < 0:
        phase += 360.0
    return 0.0 if phase >= 360.0 else phase


def response_matrix(s1, m1, s2, m2):
    """Rotation-and-scale response matrix from two excitations (original algorithm)."""
    ds_in = np.asarray(s2, float) - np.asarray(s1, float)
    ds_out = np.asarray(m2, float) - np.asarray(m1, float)
    if np.linalg.norm(ds_in) == 0:
        raise BalanceError("The two balance probe vectors must differ.")
    magnitude = np.linalg.norm(ds_out) / np.linalg.norm(ds_in)
    shift = math.atan2(ds_out[1], ds_out[0]) - math.atan2(ds_in[1], ds_in[0])
    return magnitude * np.array([[math.cos(shift), -math.sin(shift)],
                                 [math.sin(shift), math.cos(shift)]])


@dataclass
class BalanceResult:
    matrix: np.ndarray
    offset: np.ndarray
    balance_vector: np.ndarray
    ac_scale: float
    iterations: int
    converged: bool
    residual: float
    tolerance: float
    history: list = field(default_factory=list)
    message: str = ""

    @property
    def capacitance(self):
        """Original ``capacitance()``: C_s/C_ref = -vb_x * ac_scale."""
        return float(-self.balance_vector[0] * self.ac_scale)

    @property
    def dissipation(self):
        """Original ``capacitance()``: D_s/C_ref = vb_y * ac_scale."""
        return float(self.balance_vector[1] * self.ac_scale)

    @property
    def offbalance_coefficients(self):
        """Original ``offBalance()``: (Ctg, Dtg) scaling for off-balance X/Y."""
        det = float(np.linalg.det(self.matrix))
        vbx, vby = (float(v) for v in self.balance_vector)
        vdx, vdy = (float(v) for v in self.offset)
        norm = vbx ** 2 + vby ** 2
        if det == 0 or norm == 0:
            raise BalanceError("The balance result is degenerate (zero response or zero balance vector).")
        d_coeff = (vby * vdx - vbx * vdy) / norm / det * self.ac_scale
        c_coeff = -1.0 * (vbx * vdx + vby * vdy) / norm / det * self.ac_scale
        return c_coeff, d_coeff

    def offbalance(self, x, y):
        """Capacitance/dissipation (ratios) from off-balance lock-in X and Y."""
        c_coeff, d_coeff = self.offbalance_coefficients
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return c_coeff * x + d_coeff * y + self.capacitance, d_coeff * x - c_coeff * y + self.dissipation

    @property
    def magnitude(self):
        return float(math.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    @property
    def phase_deg(self):
        return float(math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0])))

    @property
    def reference_amplitude(self):
        return float(np.linalg.norm(self.balance_vector))

    @property
    def reference_phase(self):
        return vector_phase_deg(self.balance_vector)

    def as_dict(self):
        c_coeff, d_coeff = self.offbalance_coefficients
        return {
            "matrix": self.matrix.tolist(), "offset": self.offset.tolist(),
            "balance_vector": self.balance_vector.tolist(), "ac_scale": self.ac_scale,
            "capacitance_ratio": self.capacitance, "dissipation_ratio": self.dissipation,
            "c_coefficient": c_coeff, "d_coefficient": d_coeff,
            "iterations": self.iterations, "converged": self.converged,
            "residual_V": self.residual, "tolerance_V": self.tolerance, "message": self.message,
        }


class LinearBridge:
    """Iterative linear balancing of a two-arm capacitance bridge.

    ``excite(vector)`` sets the reference excitation (amplitude and relative
    phase) and ``measure()`` returns the settled lock-in (X, Y). Both are
    supplied by the caller, so the same algorithm drives real or simulated
    instruments. ``keep_running()`` is polled before every hardware step.
    """

    def __init__(self, excite, measure, *, s1=(0.1, 0.1), s2=(-0.1, -0.1), tolerance=0.015e-3,
                 iterations=10, ac_scale=1.0, max_amplitude=1.0,
                 min_amplitude=AWG_MIN_AMPLITUDE_VPP, log=None, keep_running=None):
        self._excite = excite
        self._measure = measure
        self.s1 = np.asarray(s1, dtype=float).reshape(2)
        self.s2 = np.asarray(s2, dtype=float).reshape(2)
        self.tolerance = float(tolerance)
        self.iterations = int(iterations)
        self.ac_scale = float(ac_scale)
        self.max_amplitude = float(max_amplitude)
        self.min_amplitude = float(min_amplitude)
        self._log = log or (lambda message: None)
        self._keep_running = keep_running or (lambda: True)
        if not (self.tolerance > 0 and self.iterations >= 1 and self.max_amplitude > 0):
            raise ValueError("Balance tolerance, iterations and amplitude limit must be positive.")
        for vector in (self.s1, self.s2):
            amplitude = np.linalg.norm(vector)
            if not np.all(np.isfinite(vector)) or not self.min_amplitude <= amplitude:
                raise ValueError(f"Probe vector {vector.tolist()} has amplitude {amplitude:g} V, "
                                 f"below the AWG minimum of {self.min_amplitude:g} V.")
            if not np.all(np.abs(vector) < self.max_amplitude):
                raise ValueError(f"Probe vector {vector.tolist()} exceeds the {self.max_amplitude:g} V limit.")
        self.history = []

    def _check(self):
        if not self._keep_running():
            raise BalanceCancelled("Bridge balance stopped.")

    def excite_and_measure(self, vector):
        self._check()
        self._excite(np.asarray(vector, dtype=float))
        self._check()
        measured = np.asarray(self._measure(), dtype=float).reshape(2)
        if not np.all(np.isfinite(measured)):
            raise BalanceError("The lock-in returned a non-finite reading during balancing.")
        self.history.append((np.asarray(vector, float).tolist(), measured.tolist()))
        self._log(f"  excite |s|={np.linalg.norm(vector):.4g} V, phase={vector_phase_deg(vector):.2f} deg"
                  f" -> X={measured[0]:.4e} V, Y={measured[1]:.4e} V")
        return measured

    def balance(self, matrix=None, offset=None, start=None):
        """Measure the response matrix (unless given) and iterate to balance."""
        self.history = []
        if matrix is None:
            self._log("Measuring the bridge response matrix with two probe excitations.")
            m1 = self.excite_and_measure(self.s1)
            m2 = self.excite_and_measure(self.s2)
            matrix = response_matrix(self.s1, m1, self.s2, m2)
            start_vector, start_measurement = self.s1, m1
        else:
            matrix = np.asarray(matrix, dtype=float)
            start_vector = np.asarray(start, dtype=float)
            start_measurement = self.excite_and_measure(start_vector)
        if abs(np.linalg.det(matrix)) < 1e-300:
            raise BalanceError("The bridge shows no response to the reference excitation. "
                               "Check the AWG reference output, attenuators and lock-in input.")
        self._log(f"Response |M| = {math.hypot(matrix[0, 0], matrix[1, 0]):.4g}, "
                  f"arg M = {math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])):.2f} deg")
        constant = (start_measurement - matrix @ start_vector) if offset is None else np.asarray(offset, float)
        inverse = np.linalg.inv(matrix)
        iteration, residual, converged = 1, math.inf, False
        vector = start_vector
        while True:
            vector = -inverse @ constant
            if not np.all(np.abs(vector) < self.max_amplitude):
                raise BalanceError(
                    f"Balance point {np.round(vector, 6).tolist()} V is out of range "
                    f"(|component| must stay below {self.max_amplitude:g} V). "
                    "Remove attenuators from the reference or reduce the sample excitation.")
            if np.linalg.norm(vector) < self.min_amplitude:
                raise BalanceError(
                    f"Balance amplitude {np.linalg.norm(vector):.3g} V is below the AWG minimum "
                    f"{self.min_amplitude:g} V. Add reference attenuation or raise the sample excitation.")
            measured = self.excite_and_measure(vector)
            constant = measured - matrix @ vector
            iteration += 1
            residual = float(np.linalg.norm(measured))
            self._log(f"Iteration {iteration - 1}: residual |m_b| = {residual:.3e} V "
                      f"(tolerance {self.tolerance:.3e} V)")
            if residual < self.tolerance:
                converged = True
                message = "Balanced."
                break
            if iteration > self.iterations:
                message = f"Hit the maximum number of iterations ({self.iterations}); residual {residual:.3e} V."
                break
        result = BalanceResult(matrix=matrix, offset=constant, balance_vector=vector,
                               ac_scale=self.ac_scale, iterations=iteration - 1,
                               converged=converged, residual=residual,
                               tolerance=self.tolerance, history=list(self.history), message=message)
        self._log(f"{message} C/C_ref = {result.capacitance:.6g}, D/C_ref = {result.dissipation:.6g}")
        return result

    def rebalance(self, previous):
        """Re-balance from the previous balance point, reusing its response matrix."""
        self._log("Rebalancing with the stored response matrix.")
        return self.balance(matrix=previous.matrix, start=previous.balance_vector)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulatedSample:
    """Deterministic, smooth sample response in units of C_ref (C - iD).

    A gap opens at charge neutrality with |p0| (bilayer-like). For a gate
    capacitance it appears as a dip; for penetration field capacitance as a peak.
    """

    def __init__(self, kind="gate", delta=0.0):
        self.kind = kind
        self.delta = float(delta)

    def complex_capacitance(self, vt, vb, vs):
        n0, p0 = density_field(vt, vb, vs, delta=self.delta)
        neutral = 0.05 * p0
        gap = 0.15 + 0.35 * math.tanh(abs(p0) / 1.5)
        width = 0.08 + 0.04 * abs(p0)
        profile = math.exp(-0.5 * ((n0 - neutral) / width) ** 2)
        oscillation = 0.01 * math.cos(2 * math.pi * n0 / 0.35) * math.exp(-abs(n0) / 2.0)
        if self.kind == "penetration":
            capacitance = 0.3 + 0.5 * gap * profile + 0.3 * oscillation
        else:
            capacitance = 0.8 * (1 - gap * profile) + 0.02 * math.tanh(n0) + oscillation
        dissipation = 0.18 * gap * math.exp(-0.5 * ((n0 - neutral) / (0.7 * width)) ** 2)
        return complex(capacitance, -dissipation)


class SimulatedBridgeSignal:
    """Lock-in input of a bridge referenced to the sample excitation (AWG CH1).

    V = G e^{i theta} (C_s V_s + C_ref V_r) / (C_s + C_ref + C_par), with the
    sample excitation V_s = A1 10^(-att_s/20) and the reference
    V_r = A2 10^(-att_r/20) exp(i (phi2 - phi1)). This is linear in the balance
    vector, as the original algorithm assumes, so balancing converges to
    C_s/C_ref exactly as with the real bridge.
    """

    def __init__(self, excitation, electrodes, attenuation, *, sample=None, gain=1.0,
                 phase_offset_deg=27.0, parasitic=50.0, noise_rms=0.0, seed=None):
        self._excitation = excitation
        self._electrodes = electrodes
        self._attenuation = attenuation
        self.sample = sample or SimulatedSample()
        self.gain = float(gain)
        self.phase_offset = math.radians(phase_offset_deg)
        self.parasitic = float(parasitic)
        self.noise_rms = float(noise_rms)
        self._rng = np.random.default_rng(seed)

    def __call__(self, time_constant=0.1):
        state = self._excitation()
        sample_atten, ref_atten = self._attenuation()
        vt, vb, vs = self._electrodes()
        cs = self.sample.complex_capacitance(vt, vb, vs)
        a1 = state["ch1_amplitude"] if state.get("ch1_enabled", True) else 0.0
        a2 = state["ch2_amplitude"] if state.get("ch2_enabled", True) else 0.0
        relative = math.radians(state.get("ch2_phase", 0.0) - state.get("ch1_phase", 0.0))
        v_sample = a1 * 10 ** (-sample_atten / 20.0)
        v_reference = a2 * 10 ** (-ref_atten / 20.0) * complex(math.cos(relative), math.sin(relative))
        node = (cs * v_sample + 1.0 * v_reference) / (cs + 1.0 + self.parasitic)
        value = self.gain * complex(math.cos(self.phase_offset), math.sin(self.phase_offset)) * node
        if self.noise_rms > 0:
            scale = self.noise_rms * math.sqrt(0.1 / max(float(time_constant), 1e-5))
            value += complex(*self._rng.normal(0, scale, 2))
        return value
